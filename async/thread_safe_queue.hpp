#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

// A threadsafe-queue.
template <class T>
class thread_safe_queue_t {
    std::queue<T> queue;
    mutable std::mutex mutex;
    std::condition_variable condition_variable;

   public:
    thread_safe_queue_t() : queue{}, mutex{}, condition_variable{} {}

    ~thread_safe_queue_t() {}

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
        std::unique_lock<std::mutex> lock(mutex);
        return this->queue.empty();
    }

    std::size_t size() const {
        std::unique_lock<std::mutex> lock(mutex);
        return this->queue.size();
    }
};
