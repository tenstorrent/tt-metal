// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <mutex>

namespace models::demos::deepseek_v3_b1::pipeline_manager {

// Mutex-based bounded queue for IS <-> PM communication.
// Fixed-size ring buffer, no dynamic allocation. Safe for MPSC or SPSC use.
// Not lock-free — acceptable for the API path (not per-tick hot path).
template <typename T, int Capacity>
struct BoundedQueue {
    static_assert(Capacity > 0);

    std::array<T, Capacity> buffer{};
    int head = 0;
    int tail = 0;
    int count = 0;
    mutable std::mutex mtx;

    bool try_push(const T& item) {
        std::lock_guard<std::mutex> lock(mtx);
        if (count >= Capacity) {
            return false;
        }
        buffer[head] = item;
        head = (head + 1) % Capacity;
        ++count;
        return true;
    }

    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mtx);
        if (count == 0) {
            return false;
        }
        item = buffer[tail];
        tail = (tail + 1) % Capacity;
        --count;
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx);
        return count == 0;
    }

    int size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return count;
    }
};

}  // namespace models::demos::deepseek_v3_b1::pipeline_manager
