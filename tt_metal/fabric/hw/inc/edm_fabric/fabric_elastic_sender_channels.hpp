// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_outbound_sender_channel_interface.hpp"

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

template <size_t CHUNK_N_PKTS>
struct ElasticSenderChannel : public SenderEthChannelInterface<ElasticSenderChannel<CHUNK_N_PKTS>> {
    using chunk_t = EthChannelBuffer<PACKET_HEADER_TYPE, CHUNK_N_PKTS>;
    using chunk_iterator_t tt::tt_fabric::OnePassIteratorStaticSizes<uint32_t, CHUNK_N_PKTS, PACKET_SIZE_BYTES / sizeof(uint32_t)>;

    // A container for all the chunks in use by this sender channel.
    CircularBuffer<chunk_t*, REQUESTED_SIZE> open_chunks_window;
    
    // This iterator is responsible for iterating through the "leading" sender channel
    // chunk and advances as new packets are sent over ethernet (i.e. in `send_next_data`)
    chunk_iterator_t sending_chunk_iterator;

    // This iterator is responsible for iterating through the "trailing" sender channel
    // chunk and advances as completion acks are received.
    chunk_iterator_t completion_chunk_iterator;


    std::size_t cached_next_buffer_slot_addr;


    FORCE_INLINE void init_impl(
        size_t channel_base_address, size_t max_eth_payload_size_in_bytes, size_t header_size_bytes) {
        static_assert(false, "Unimplemented");
    }

    FORCE_INLINE size_t get_cached_next_buffer_slot_addr_impl() const {
        return this->cached_next_buffer_slot_addr;
        static_assert(false, "Unimplemented");
    }

    FORCE_INLINE void advance_to_next_cached_buffer_slot_addr_impl() {
        static_assert(false, "Unimplemented");
        sending_chunk_iterator.increment();
        if (sending_chunk_iterator.is_done()) {
            
        }
    }
};