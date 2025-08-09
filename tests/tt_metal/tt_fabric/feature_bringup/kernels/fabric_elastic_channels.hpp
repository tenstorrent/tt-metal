// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"

#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>

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

template <typename T, size_t MAX_SIZE>
struct Stack {
    static constexpr size_t REAL_SIZE = MAX_SIZE + 1;
    using top_idx_t = int8_t;
    static constexpr top_idx_t EMPTY_VALUE = 0;
    static_assert(REAL_SIZE < std::numeric_limits<top_idx_t>::max());
    std::array<T, REAL_SIZE> data;
    top_idx_t top;

    Stack() : top(EMPTY_VALUE) {}

    FORCE_INLINE bool is_empty() const { return top == EMPTY_VALUE; }

    FORCE_INLINE bool is_full() const {
        // Should only need calling for validation/debugging purposes
        return top == REAL_SIZE - 1;
    }

    FORCE_INLINE void push(T value) {
        ASSERT(!is_full());
        data[top] = value;
        top++;
    }

    FORCE_INLINE T pop() {
        ASSERT(!is_empty());
        T value = data[top];
        top--;
        return value;
    }

    FORCE_INLINE T peek() const {
        ASSERT(!is_empty());
        return data[top];
    }
};

// around 40% faster on WH compared to the default implementation
// pop about (14 vs 12 cycles) 20% faster, push about (16 vs 10 cycles) 40% faster
// not an exact cycle count given how timestamping works
template <typename T, size_t MAX_SIZE>
struct WormholeEfficientStack {
    static constexpr size_t REAL_SIZE = MAX_SIZE + 1;
    using top_idx_t = int8_t;
    static constexpr top_idx_t EMPTY_VALUE = 0;
    static_assert(REAL_SIZE < std::numeric_limits<top_idx_t>::max());
    std::array<T, REAL_SIZE> data;
    T* top;

    WormholeEfficientStack() : data({}), top(&data[0]) {}

    FORCE_INLINE bool is_empty() const { return top == &data[0]; }

    FORCE_INLINE bool is_full() const {
        // Should only need calling for validation/debugging purposes
        return top == &data[REAL_SIZE - 1];
    }

    FORCE_INLINE void push(T value) {
        ASSERT(!is_full());
        *top = value;
        top++;
    }

    FORCE_INLINE T pop() {
        ASSERT(!is_empty());
        T value = *top;
        --top;
        return value;
    }

    FORCE_INLINE T peek() const {
        ASSERT(!is_empty());
        return *top;
    }
};

template <size_t N_CHUNKS, size_t CHUNK_N_PKTS>
struct ChannelBuffersPool {
    using chunk_t = EthChannelBuffer<CHUNK_N_PKTS>;
    using free_chunks_stack_t =
#if defined(ARCH_WORMHOLE)
        WormholeEfficientStack<chunk_t*, N_CHUNKS>;
#else
        Stack<chunk_t*, N_CHUNKS>;
#endif

    std::array<chunk_t, N_CHUNKS> all_buffers;
    free_chunks_stack_t free_chunks;

    // using init_regions_t = std::initializer_list<std::tuple<size_t, uint8_t>>;
    using init_regions_t = std::array<std::tuple<size_t, uint8_t>, N_CHUNKS>;

    void init(const init_regions_t& buffer_regions, size_t buffer_size_bytes, size_t header_size_bytes) {
        size_t idx = 0;
        for (const auto& [channel_base_address, channel_base_id] : buffer_regions) {
            ASSERT(idx < N_CHUNKS);
            new (&all_buffers[idx]) chunk_t(channel_base_address, buffer_size_bytes, header_size_bytes, N_CHUNKS);
            free_chunks.push(&all_buffers[idx]);
            idx++;
        }
    }

    FORCE_INLINE chunk_t* get_free_chunk() {
        ASSERT(!free_chunks.is_empty());
        return free_chunks.pop();
    }

    FORCE_INLINE void return_chunk(chunk_t* chunk) { free_chunks.push(chunk); }

    FORCE_INLINE bool is_empty() const {
        auto empty = free_chunks.is_empty();
        return empty;
    }

    FORCE_INLINE bool is_full() const { return free_chunks.is_full(); }
};

template <typename T, size_t REQUESTED_SIZE>
struct CircularBuffer {
    // we pad up to the next power of 2 to get nicer arithmetic
    static constexpr size_t CAPACITY = get_next_power_of_2(REQUESTED_SIZE);
    static_assert(is_power_of_2(CAPACITY), "CAPACITY must be a power of 2");

    std::array<T, CAPACITY> data;

    ChannelBufferPointer<CAPACITY> wr_ptr;
    ChannelBufferPointer<CAPACITY> rd_ptr;

    CircularBuffer() = default;

    FORCE_INLINE void push(const T value) {
        data[wr_ptr.get_buffer_index()] = value;
        wr_ptr.increment();
    }

    FORCE_INLINE T pop() {
        T value = data[rd_ptr.get_buffer_index()];
        rd_ptr.increment();
        return value;
    }

    FORCE_INLINE T peek_front() const {
        // The rd_ptr points to the oldest element in the CB (aka the front since we are
        // pushing to the "back")
        return data[rd_ptr.get_buffer_index()];
    }

    FORCE_INLINE bool is_empty() const { return rd_ptr.is_caught_up_to(wr_ptr); }

    FORCE_INLINE bool is_full() const { return wr_ptr.distance_behind(rd_ptr) == 1; }
};

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

}  // namespace tt::tt_fabric
