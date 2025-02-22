// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/thread_pool.hpp>

ThreadPool::WorkerQueue::WorkerQueue() {
    // Initialize ring buffer for traversal. Each node points to the subsequent node, except for the last one, which
    // points to the head.
    for (int node_idx = 0; node_idx < ring_buffer_size; node_idx++) {
        (node_idx < ring_buffer_size - 1) ? ring_buffer[node_idx].next = (&ring_buffer[node_idx + 1])
                                          : ring_buffer[node_idx].next = &(ring_buffer[0]);
    }
    // Initialize head and tail ptrs to start of ring buffer.
    this->head = ring_buffer;
    this->tail = ring_buffer;
}

void ThreadPool::WorkerQueue::push(std::packaged_task<void()>&& task) {
    // Stall condition: this push will update the tail (wptr)
    // to match the location of head (rptr). The current push can
    // thus overwrite data that's being read. Stall until head
    // has progressed (data has been read).
    // A stall is only required when the ring_buffer backing the queue
    // is full. Realistically, this should never happen, given the size
    while (tail.load()->next == head.load());
    tail.load()->data = std::move(task);
    tail.store(tail.load()->next);
}

std::packaged_task<void()>&& ThreadPool::WorkerQueue::pop() {
    ThreadPool::WorkerQueue::Node* oldHead = pop_head();
    return std::move(oldHead->data);
}

bool ThreadPool::WorkerQueue::empty() const { return head.load() == tail.load(); }

ThreadPool::WorkerQueue::Node* ThreadPool::WorkerQueue::pop_head() {
    ThreadPool::WorkerQueue::Node* oldHead = head.load();
    if (oldHead == tail.load()) {
        return nullptr;  // Queue is empty
    }
    head.store(oldHead->next);
    return oldHead;
}

ThreadPool::ThreadPool(size_t thread_count) : shutdown_(false) {
    workers_.reserve(thread_count);

    for (size_t i = 0; i < thread_count; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::packaged_task<void()> task;  // Task container for this thread
                {
                    task_semaphore.acquire();  // Ensures 1:1 task to worker mapping
                    if (shutdown_) {
                        return;
                    }
                    // The lock free queue only allows a single reader/single writer
                    // With multiple readers, we must use a lock to synchronize
                    std::unique_lock<std::mutex> lock(mutex_);
                    task = std::move(tasks_.pop());  // Move the packaged_task
                }
                task();     // Execute the packaged_task
                counter--;  // Notify maibn thread that a task was completed
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    shutdown_ = true;
    // Notify all workers that a shutdown signal was sent
    for (size_t i = 0; i < workers_.size(); ++i) {
        task_semaphore.release();
    }
    for (std::thread& worker : workers_) {
        worker.join();
    }
}

void ThreadPool::barrier() const noexcept { while (counter); }

std::size_t ThreadPool::num_threads() const noexcept { return workers_.size(); }
