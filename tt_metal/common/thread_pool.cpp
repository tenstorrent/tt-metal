// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/asio.hpp>
#include <future>
#include <iostream>
#include <numa.h>
#include <semaphore>

#include "tt_metal/common/thread_pool.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal {

std::unordered_map<int, std::vector<uint32_t>> get_cpu_cores_per_numa_node() {
    std::unordered_map<int, std::vector<uint32_t>> cpu_cores_per_numa_node = {};
    for (int cpu = 0; cpu < numa_num_configured_cpus(); ++cpu) {
        int node = numa_node_of_cpu(cpu);
        cpu_cores_per_numa_node[node].push_back(cpu);
    }
    return cpu_cores_per_numa_node;
}

namespace threading_primitives {

// Data Structure used to queue and submit tasks to custom thread-pool backends.
// Implemented as a statically allocated ring buffer that holds a task in each slot.
class TaskQueue {
public:
    TaskQueue() {
        // Initialize ring buffer for traversal. Each node points to the subsequent node, except for the last one,
        // which points to the head.
        for (int node_idx = 0; node_idx < ring_buffer_size_; node_idx++) {
            (node_idx < ring_buffer_size_ - 1) ? ring_buffer_[node_idx].next = (&ring_buffer_[node_idx + 1])
                                               : ring_buffer_[node_idx].next = &(ring_buffer_[0]);
        }
        // Initialize head and tail ptrs to start of ring buffer.
        head_ = ring_buffer_;
        tail_ = ring_buffer_;
    }
    // Push task to queue (writer).
    void push(std::function<void()>&& task) {
        // Stall condition: this push will update the tail (wptr)
        // to match the location of head (rptr). The current push can
        // thus overwrite data that's being read. Stall until head
        // has progressed (data has been read).
        // A stall is only required when the ring_buffer_ backing the queue
        // is full. Realistically, this should never happen, given the size
        while (tail_.load()->next == head_.load());
        tail_.load()->data = std::move(task);
        tail_.store(tail_.load()->next);
    }
    // Pop task from queue (reader).
    std::function<void()>&& pop() {
        TaskQueue::Node* old_head = pop_head();
        return std::move(old_head->data);
    }

private:
    // Node object, representing a slot in the queue.
    struct Node {
        std::function<void()> data;
        Node* next = nullptr;
    };
    // Read and write pointers for managing the queue.
    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;

    Node* pop_head() {
        Node* old_head = head_.load();
        if (old_head == tail_.load()) {
            TT_THROW("Cannot pop tasks from an empty queue.");
            return nullptr;  // Queue is empty
        }
        head_.store(old_head->next);
        return old_head;
    }
    // Statically allocated ring buffer containing
    // node objects, which contain handles to data
    // and another node object to traverse ring buffer.
    const static uint32_t ring_buffer_size_ = 8192;
    Node ring_buffer_[ring_buffer_size_];
};
// NUMA + CPU Affinity aware executor, used by custom thread-pool implementations.
// Contains:
//  1. A TaskQueue where tasks can be submitted by the user, to be asynchronously executed
//  2. A worker thread to asynchronously execute tasks
//  3. Primitves to synchronize the application and worker thread
// Usage:
// This executor should only be used to asynchronously process tasks for a specific TT-Device
// (specified through the physical_device_id constructor argument).
// The executor is NUMA aware, i.e. it will bind its worker thread to a NUMA node that is "closest"
// to its physical device. The logical_cpu_offset constructor argument can be used to specify the
// logical base offset within a NUMA node when binding the worker thread.
// The CPU selection algorithm is:
// CPUs[numa_node][(physical_device_id + logical_cpu_offset) / num_cores_on_numa_node]
class NumaAwareExecutor {
public:
    NumaAwareExecutor(uint32_t physical_device_id, uint32_t logical_cpu_offset) : tasks_() {
        static std::unordered_map<int, std::vector<uint32_t>> cpu_cores_per_numa_node = get_cpu_cores_per_numa_node();

        worker = std::thread([this]() {
            std::function<void()> task;  // Task container for this thread
            while (true) {
                {
                    task_semaphore_.acquire();
                    if (shutdown_) {
                        return;
                    }
                    task = std::move(tasks_.pop());
                }
                task();
                // Atomically decrement counter used to synchronize with main thread
                // and notify the main thread if all tasks have completed
                if (counter_.fetch_sub(1, std::memory_order_release) == 1) {
                    counter_.notify_all();
                }
            }
        });
        auto numa_node = tt::Cluster::instance().get_numa_node_for_device(physical_device_id);
        auto& cpu_cores_on_node = cpu_cores_per_numa_node[numa_node];
        auto cpu_core_for_worker =
            cpu_cores_on_node[(physical_device_id + logical_cpu_offset) % cpu_cores_on_node.size()];
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_core_for_worker, &cpuset);
        int rc = pthread_setaffinity_np(worker.native_handle(), sizeof(cpu_set_t), &cpuset);
        if (rc) {
            log_warning(
                tt::LogMetal,
                "Unable to bind worker thread to CPU Core. May see performance degradation. Error Code: {}",
                rc);
        }
    }

    ~NumaAwareExecutor() {
        // Destructor called in main thread.
        // Wait to ensure that the worker thread has completed.
        this->wait();
        shutdown_ = true;
        task_semaphore_.release();
        worker.join();
    }

    void enqueue(std::function<void()>&& f) {
        tasks_.push(std::move(f));  // Move the task directly into queue
        task_semaphore_.release();  // Notify a worker that a task is available
        // Light-Weight counter increment to track the number of tasks in flight
        // Need this because a counting_semaphore does not allow querying state
        counter_++;
    }

    void wait() const {
        // Wait until all tasks have completed (counter_ == 0)
        // To avoid spinning, sleep until notified by the worker threads
        // or counter_ changes (this only happens with a spurious wakeup)
        int current;
        while ((current = counter_.load(std::memory_order_acquire)) > 0) {
            counter_.wait(current, std::memory_order_relaxed);
        }
    }

private:
    TaskQueue tasks_;
    std::thread worker;
    std::atomic<int> counter_ = 0;
    std::counting_semaphore<> task_semaphore_{0};
    bool shutdown_ = false;
};

}  // namespace threading_primitives

namespace thread_pool_impls {
// Implementations conforming to the ThreadPool interface.
using threading_primitives::NumaAwareExecutor;

// Boost backed thread-pool.
class BoostThreadPool : public ThreadPool {
public:
    BoostThreadPool(size_t thread_count) : pool_(thread_count) {
        // Given the current use case, we don't expect to
        // enqueue more tasks than the number of threads.
        // Add a factor of safety and modify as needed.
        futures_.reserve(thread_count * 4);
    }

    ~BoostThreadPool() noexcept override = default;

    void enqueue(std::function<void()>&& f, uint32_t thread_idx) override {
        std::packaged_task<void()> task(std::move(f));
        futures_.push_back(task.get_future());
        boost::asio::post(pool_, [executor = std::move(task)]() mutable { executor(); });
    }

    void wait() override {
        for (auto& future : futures_) {
            future.get();
        }
        futures_.clear();
    }

private:
    boost::asio::thread_pool pool_;
    std::vector<std::future<void>> futures_;
};

// Custom Thread-Pool using the threading::Executor class.
// Allows enqueuing tasks tied to specific devices.
class DeviceBoundThreadPool : public ThreadPool {
public:
    DeviceBoundThreadPool(uint32_t thread_count, uint32_t logical_cpu_offset) {
        workers_.reserve(thread_count);
        for (uint32_t i = 0; i < thread_count; i++) {
            workers_.emplace_back(std::make_unique<NumaAwareExecutor>(i, logical_cpu_offset));
        }
    }

    void enqueue(std::function<void()>&& f, uint32_t thread_idx) override {
        workers_[thread_idx]->enqueue(std::move(f));
    }

    void wait() override {
        for (auto& worker : workers_) {
            worker->wait();
        }
    }

private:
    std::vector<std::unique_ptr<NumaAwareExecutor>> workers_;
};

}  // namespace thread_pool_impls

std::shared_ptr<ThreadPool> create_boost_thread_pool(int num_threads) {
    return std::make_shared<thread_pool_impls::BoostThreadPool>(num_threads);
}

std::shared_ptr<ThreadPool> create_device_bound_thread_pool(int num_threads, uint32_t logical_cpu_offset) {
    return std::make_shared<thread_pool_impls::DeviceBoundThreadPool>(num_threads, logical_cpu_offset);
}

}  // namespace tt::tt_metal
