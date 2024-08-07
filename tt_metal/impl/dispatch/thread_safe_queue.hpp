// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

namespace tt {
namespace tt_metal {

// A threadsafe-queue.
template <class T>
class thread_safe_queue_t {
    std::queue<T> queue;
    mutable std::mutex mutex;
    std::condition_variable condition_variable;

   public:
    thread_safe_queue_t() : queue{}, mutex{}, condition_variable{} {}

    thread_safe_queue_t(const thread_safe_queue_t& other) = delete;
    thread_safe_queue_t& operator=(const thread_safe_queue_t&& other) = delete;

    thread_safe_queue_t(thread_safe_queue_t&& other) {
        std::lock_guard<std::mutex> lock(mutex);
        queue = std::move(other.queue);
    }

    thread_safe_queue_t& operator=(thread_safe_queue_t&& other) {
        std::lock_guard<std::mutex> lock(mutex);
        queue = std::move(other.queue);
        return *this;
    }

    void emplace_back(T&& element) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.emplace(element);
        condition_variable.notify_one();
    }

    T pop_front(void) {
        std::unique_lock<std::mutex> lock(mutex);
        while (queue.empty()) {
            condition_variable.wait(lock);
        }
        T element = queue.front();
        queue.pop();
        return element;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex);
        return this->queue.empty();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex);
        return this->queue.size();
    }
};
}  // namespace tt_metal
}  // namespace tt
