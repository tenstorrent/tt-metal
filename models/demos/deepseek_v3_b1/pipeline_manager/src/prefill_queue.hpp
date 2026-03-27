// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <mutex>

namespace models::demos::deepseek_v3_b1::pipeline_manager {

// Bounded session queue for users in PREFILL state.
// API pushes on submit/continue, Writer pops/rotates for chunked round-robin scheduling.
// The mutex is only contended on submit (rare, not on the per-tick hot path).
struct PrefillQueue {
    std::deque<int32_t> queue;
    std::mutex mtx;

    void push(int uid) {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push_back(static_cast<int32_t>(uid));
    }

    bool try_front(int& uid) {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.empty()) {
            return false;
        }
        uid = queue.front();
        return true;
    }

    void pop_front() {
        std::lock_guard<std::mutex> lock(mtx);
        if (!queue.empty()) {
            queue.pop_front();
        }
    }

    // Move front to back for chunked round-robin across users.
    void rotate() {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.size() > 1) {
            int32_t front = queue.front();
            queue.pop_front();
            queue.push_back(front);
        }
    }

    void remove(int uid) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = std::remove(queue.begin(), queue.end(), static_cast<int32_t>(uid));
        queue.erase(it, queue.end());
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }
};

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
