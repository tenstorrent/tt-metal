// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_outbound_sender_channel_interface.hpp"

template <typename T, size_t REQUESTED_SIZE>
struct WormholeEfficientCircularBuffer {
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

    WormholeEfficientCircularBuffer() :
        data({}), wrptr(&data[0]), rdptr(&data[0]), end(&data[CAPACITY]), end_m1(&data[CAPACITY - 1]), cnt(0) {}

    FORCE_INLINE void incr_wrap_ptr(T*& ptr) {
        // ptr++;
        // if (ptr == end) {
        //     ptr = &data[0];
        // }

        // 2x faster for release?
        bool last = ptr == end_m1;
        // ptr += last * -CAPACITY + !last;
        ptr += 1 - last * CAPACITY;
    }

    FORCE_INLINE void push(const T value) {  // acquire
        auto wrptr_old = wrptr;

        // 6.5
        // bool last = wrptr == end_m1;
        // wrptr += !last - (last * CAPACITY_M1);

        // 8.00
        // bool last = wrptr == end_m1;
        // wrptr += 1 - (last * CAPACITY);

        // 7.00
        // bool last = wrptr == end_m1;
        // wrptr += !last + (last * -CAPACITY_M1);

        // 7.00
        // bool last = wrptr == end_m1;
        // wrptr += (last * -CAPACITY_M1) + !last;

        // 8.00
        // bool last = wrptr == end_m1;
        // wrptr = wrptr - (last * CAPACITY) + 1;

        // 6.00
        bool last = wrptr == end_m1;
        wrptr = wrptr - (last * CAPACITY_M1) + !last;

        cnt++;
        *wrptr_old = value;
    }

    FORCE_INLINE T pop() {  // release
        // T value = *rdptr;
        // incr_wrap_ptr(rdptr);
        // cnt--;
        // return value;
        auto rdptr_old = rdptr;

        // 9.5
        bool last = rdptr == end_m1;
        rdptr += 1 - last * CAPACITY;

        // 9.5
        // bool last = rdptr == end_m1;
        // rdptr = rdptr - last * CAPACITY + 1;

        // 9.00 (negatively affects push performance - from the store??? - up to 8.00)
        // bool last = rdptr == end_m1;
        // rdptr = rdptr - last * CAPACITY_M1 + !last;

        // 9.50
        // bool last = rdptr == end_m1;
        // rdptr = rdptr - last * CAPACITY + 1;

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
