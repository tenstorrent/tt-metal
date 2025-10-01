// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_attribs.h"
#include "utils/utils.h"   // for is_power_of_2

namespace tt::tt_fabric {

constexpr bool is_next_power_of_2(size_t n) { return n > 0 && (n & (n - 1)) == 0; }
constexpr size_t get_next_power_of_2(size_t n) {
    if (n <= 1) {
        return 1;
    }
    if (is_power_of_2(n)) {
        return n;
    }
    size_t power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

template <typename T, size_t REQUESTED_SIZE>
struct CircularBuffer {
    // we pad up to the next power of 2 to get nicer arithmetic
    static constexpr size_t CAPACITY = get_next_power_of_2(REQUESTED_SIZE);
    static constexpr size_t CAPACITY_M1 = CAPACITY - 1;
    static_assert(is_power_of_2(CAPACITY), "CAPACITY must be a power of 2");

    std::array<T, CAPACITY> data;
    T* wrptr;
    T* rdptr;
    T* end;
    T* end_m1;
    uint8_t cnt;

    CircularBuffer() :
        data({}), wrptr(&data[0]), rdptr(&data[0]), end(&data[CAPACITY]), end_m1(&data[CAPACITY - 1]), cnt(0) {}

    FORCE_INLINE void incr_wrap_ptr(T*& ptr) {
        bool last = ptr == end_m1;
        ptr += 1 - last * CAPACITY;
    }

    FORCE_INLINE void push(const T value) {  // acquire
        auto wrptr_old = wrptr;

        bool last = wrptr == end_m1;
        wrptr = wrptr - (last * CAPACITY_M1) + !last;

        cnt++;
        *wrptr_old = value;
    }

    FORCE_INLINE T pop() {  // release
        auto rdptr_old = rdptr;

        bool last = rdptr == end_m1;
        rdptr += 1 - last * CAPACITY;

        cnt--;
        return *rdptr_old;
    }

    FORCE_INLINE T peek_front() const {
        // The rd_ptr points to the oldest element in the CB (aka the front since we are
        // pushing to the "back")
        return *rdptr;
    }

    FORCE_INLINE bool is_empty() const { return cnt == 0; }

    FORCE_INLINE bool is_full() const { return cnt == CAPACITY; }
};

}  // namespace tt::tt_fabric