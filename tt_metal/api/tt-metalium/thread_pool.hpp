// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thread>
#include <mutex>
#include <semaphore>
#include <functional>
#include <future>
#include <memory>

namespace tt::tt_metal {

class ThreadPool {
public:
    explicit ThreadPool(size_t thread_count = std::thread::hardware_concurrency());
    ~ThreadPool();

    template <class F>
    void enqueue(F&& f) {
        tasks_.push(std::move(f));     // Move the task directly into queue
        task_semaphore_.release();     // Notify a worker that a task is available
        // Light-Weight counter increment to track the number of tasks in flight
        // Need this because a counting_semaphore does not allow querying state
        counter_++;
    }
    void barrier() const noexcept;

    std::size_t num_threads() const noexcept;

private:
    class WorkerQueue {
    public:
        WorkerQueue();
        void push(std::function<void()>&& task);
        std::function<void()>&& pop();
        bool empty() const;

    private:
        struct Node {
            std::function<void()> data;
            Node* next = nullptr;
        };

        std::atomic<Node*> head_;
        std::atomic<Node*> tail_;

        Node* pop_head();
        // Statically allocated ring buffer containing
        // node objects, which contain handles to data
        // and another node object to traverse ring buffer.
        const static uint32_t ring_buffer_size_ = 32768;
        Node ring_buffer_[ring_buffer_size_];
    };

    // Worker threads backing the pool
    std::vector<std::thread> workers_;
    // Task queue
    WorkerQueue tasks_;
    // Mutex to synchronize workers when reading
    // from task queue
    std::mutex mutex_;
    // Counting Semaphore used by main thread to
    // notify workers when a task is available
    std::counting_semaphore<> task_semaphore_{0};
    // Atomic counter used by workers to notify
    // main thread when all tasks are complete
    std::atomic<int> counter_ = 0;
    bool shutdown_;
};

}  // namespace tt::tt_metal
