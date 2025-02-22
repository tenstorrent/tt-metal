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

class ThreadPool {
public:
    explicit ThreadPool(size_t thread_count = std::thread::hardware_concurrency());
    ~ThreadPool();

    template <class F>
    void enqueue(F&& f) {
        // Bind the function to a packaged_task
        auto task = std::packaged_task<void()>(std::forward<F>(f));
        tasks_.push(std::move(task));  // Move the task directly into queue
        task_semaphore.release();      // Notify a worker that a task is available
        // Light-Weight counter increment to track the number of tasks in flight
        // Need this because a counting_semaphore does not allow querying state
        counter++;
    }
    void barrier() const noexcept;

    std::size_t num_threads() const noexcept;

private:
    class WorkerQueue {
    public:
        WorkerQueue();
        void push(std::packaged_task<void()>&& task);
        std::packaged_task<void()>&& pop();
        bool empty() const;

    private:
        struct Node {
            // Use packaged_task, since it is
            // move only. Forces us to ensure
            // that tasks are enqueued in a
            // light-weight manner
            std::packaged_task<void()> data;
            Node* next = nullptr;
        };

        std::atomic<Node*> head;
        std::atomic<Node*> tail;

        Node* pop_head();
        // Statically allocated ring buffer containing
        // node objects, which contain handles to data
        // and another node object to traverse ring buffer.
        const static uint32_t ring_buffer_size = 32768;
        Node ring_buffer[ring_buffer_size];
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
    std::counting_semaphore<> task_semaphore{0};
    // Atomic counter used by workers to notify
    // main thread when all tasks are complete
    std::atomic<int> counter = 0;
    bool shutdown_;
};
