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
        --top;
        T value = *top;
        return value;
    }

    FORCE_INLINE T peek() const {
        ASSERT(!is_empty());
        return *top;
    }
};

template <typename T, typename INHERITED_TYPE>
struct OnePassIteratorBase {
    T* current_ptr;
    T* end_ptr;

    OnePassIteratorBase() : current_ptr(nullptr), end_ptr(nullptr) {}

    FORCE_INLINE T* get_current_ptr() const { return current_ptr; }
    FORCE_INLINE void increment() { static_cast<INHERITED_TYPE*>(this)->increment_impl(); }

    FORCE_INLINE bool is_done() const { return current_ptr == end_ptr; }

    FORCE_INLINE void reset_to(T* base_ptr) { static_cast<INHERITED_TYPE*>(this)->reset_to_impl(base_ptr); }
};

template <typename T, size_t NUM_ENTRIES, size_t ENTRY_SIZE_BYTES>
struct OnePassIteratorStaticSizes
    : public OnePassIteratorBase<T, OnePassIteratorStaticSizes<T, NUM_ENTRIES, ENTRY_SIZE_BYTES>> {
    OnePassIteratorStaticSizes() :
        OnePassIteratorBase<T, OnePassIteratorStaticSizes<T, NUM_ENTRIES, ENTRY_SIZE_BYTES>>() {}

    FORCE_INLINE void increment_impl() { this->current_ptr += ENTRY_SIZE_BYTES; }

    FORCE_INLINE void reset_to_impl(T* base_ptr) {
        this->current_ptr = base_ptr;
        this->end_ptr = base_ptr + (NUM_ENTRIES * ENTRY_SIZE_BYTES);
    }
};

template <typename T>
struct OnePassIterator : public OnePassIteratorBase<T, OnePassIterator<T>> {
    uint32_t num_entries;
    uint32_t increment_size;
    OnePassIterator(uint32_t num_entries, uint32_t entry_size_bytes) :
        OnePassIteratorBase<T, OnePassIterator<T>>(), num_entries(num_entries), increment_size(entry_size_bytes) {}

    FORCE_INLINE void increment_impl() { this->current_ptr += increment_size; }

    FORCE_INLINE void reset_to_impl(T* base_ptr) {
        this->current_ptr = base_ptr;
        this->end_ptr = base_ptr + (num_entries * increment_size);
    }
};

template <typename T, size_t SIZE>
struct BarrelIterator {
    T* base_ptr;
    uint8_t current_idx;

    BarrelIterator(T* base_ptr) : base_ptr(base_ptr), current_idx(0) {}

    FORCE_INLINE T* get_current_ptr() const { return base_ptr + current_idx; }
    FORCE_INLINE T get_current_value() const { return *(base_ptr + current_idx); }
    FORCE_INLINE void increment() { current_idx = wrap_increment<SIZE, T>(); }
};

template <size_t N_CHUNKS, size_t CHUNK_N_PKTS>
struct ChannelBuffersPool {
    using chunk_t = EthChannelBuffer<PACKET_HEADER_TYPE, CHUNK_N_PKTS>;
    using free_chunks_stack_t =
#if defined(ARCH_WORMHOLE)
        WormholeEfficientStack<chunk_t*, N_CHUNKS>;
#else
        Stack<chunk_t*, N_CHUNKS>;
#endif

    std::array<chunk_t, N_CHUNKS> all_buffers;
    free_chunks_stack_t free_chunks;

    // using chunk_base_address_t = std::initializer_list<std::tuple<size_t, uint8_t>>;
    using chunk_base_address_t = std::array<size_t, N_CHUNKS>;

    void init(const chunk_base_address_t& buffer_regions, size_t buffer_size_bytes, size_t header_size_bytes) {
        size_t idx = 0;
        for (const auto& chunk_base_address : buffer_regions) {
            ASSERT(idx < N_CHUNKS);
            new (&all_buffers[idx]) chunk_t(chunk_base_address, buffer_size_bytes, header_size_bytes);
            free_chunks.push(&all_buffers[idx]);
            idx++;
        }
    }

    FORCE_INLINE chunk_t* get_free_chunk() {
        ASSERT(!free_chunks.is_empty());
        return free_chunks.pop();
    }

    FORCE_INLINE void return_chunk(chunk_t* chunk) { free_chunks.push(chunk); }

    FORCE_INLINE bool is_empty() const { return free_chunks.is_empty(); }

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

// Shares chunks with remote sources (workers or fabric routers)
struct FabricChunkMessageAvailableMessage {
    static constexpr uint32_t NEXT_CHUNK_VALID_FLAG = 1 << 31;
    static constexpr uint32_t NEXT_CHUNK_VALUE_MASK = NEXT_CHUNK_VALID_FLAG - 1;
    FORCE_INLINE static uint32_t pack(uint32_t chunk_base_address) {
        return chunk_base_address | NEXT_CHUNK_VALID_FLAG;
    }

    FORCE_INLINE static uint32_t unpack(uint32_t message) { return message & ~NEXT_CHUNK_VALID_FLAG; }
};

struct SenderChannelView {
    volatile uint32_t* next_chunk_ptr;

    SenderChannelView(volatile uint32_t* next_chunk_ptr) : next_chunk_ptr(next_chunk_ptr) {}

    FORCE_INLINE void wait_for_new_chunk() {
        while (!*next_chunk_ptr) {
        }
    }

    FORCE_INLINE bool has_new_chunk() {
        return *next_chunk_ptr & FabricChunkMessageAvailableMessage::NEXT_CHUNK_VALID_FLAG;
    }

    FORCE_INLINE void clear_new_chunk_flag() { *next_chunk_ptr = 0; }

    FORCE_INLINE uint32_t get_next_chunk() {
        uint32_t value = *next_chunk_ptr;
        return FabricChunkMessageAvailableMessage::unpack(value);
    }
};

// Used by the worker to know where to send packets to next
// TODO: export constants via JIT_BUILD for better performance (fewer RT args and literals)
struct WorkerFabricWriterAdapter {
    SenderChannelView sender_channel_view;
    // We have a uint8_t (byte) iterator with a step size of buffer slot. The uint8_t is the type because then we can
    // safely due byte address increments
    tt::tt_fabric::OnePassIterator<uint8_t> current_chunk;

    WorkerFabricWriterAdapter(
        volatile uint32_t* next_chunk_ptr, uint32_t num_buffer_slots_per_chunk, uint32_t max_payload_size_bytes) :
        sender_channel_view(next_chunk_ptr), current_chunk(num_buffer_slots_per_chunk, max_payload_size_bytes) {}

    FORCE_INLINE bool has_valid_destination() { return !current_chunk.is_done(); }

    FORCE_INLINE void advance_to_next_buffer_slot() { current_chunk.increment(); }

    FORCE_INLINE bool new_chunk_is_available() { return sender_channel_view.has_new_chunk(); }

    FORCE_INLINE size_t get_next_write_address() const { return (size_t)current_chunk.get_current_ptr(); }

    // return true if the new chunk was updated
    FORCE_INLINE void update_to_new_chunk() {
        //...
        ASSERT(sender_channel_view.has_new_chunk());
        ASSERT(current_chunk.is_done());
        auto chunk_base_address = sender_channel_view.get_next_chunk();
        auto new_chunk_base_address = chunk_base_address;
        sender_channel_view.clear_new_chunk_flag();
        current_chunk.reset_to(reinterpret_cast<uint8_t*>(new_chunk_base_address));
    }
};

}  // namespace tt::tt_fabric
