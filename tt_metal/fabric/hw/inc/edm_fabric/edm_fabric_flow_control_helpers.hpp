// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <limits>

#include "api/debug/assert.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/named_types.hpp"

#include "api/alignment.h"
#include "internal/risc_attribs.h"

#include "api/debug/assert.h"

namespace tt::tt_fabric {

using BufferIndex = NamedType<uint8_t, struct BufferIndexType>;
using BufferPtr = NamedType<uint8_t, struct BufferPtrType>;

// Increments val and wraps to 0 if it reaches limit
template <size_t LIMIT = 0, typename T>
FORCE_INLINE auto wrap_increment(T val) -> T {
    constexpr bool is_pow2 = LIMIT != 0 && is_power_of_2(LIMIT);
    if constexpr (LIMIT == 1) {
        return val;
    } else if constexpr (LIMIT == 2) {
        return 1 - val;
    } else if constexpr (is_pow2) {
        return (val + 1) & (static_cast<T>(LIMIT - 1));
    } else {
        return (val == static_cast<T>(LIMIT - 1)) ? static_cast<T>(0) : static_cast<T>(val + 1);
    }
}

// Increments val and wraps to 0 if it reaches limit
template <typename T>
FORCE_INLINE auto wrap_increment(T val, T limit) -> T {
    return (val == static_cast<T>(limit - 1)) ? static_cast<T>(0) : static_cast<T>(val + 1);
}

template <size_t LIMIT, typename T>
FORCE_INLINE auto wrap_increment_n(T val, uint8_t increment) -> T {
    constexpr bool is_pow2 = LIMIT != 0 && is_power_of_2(LIMIT);
    if constexpr (LIMIT == 1) {
        return val;
    } else if constexpr (LIMIT == 2) {
        return 1 - val;
    } else if constexpr (is_pow2) {
        return (val + increment) & (LIMIT - 1);
    } else {
        T new_unadjusted_val = val + increment;
        bool wraps = new_unadjusted_val >= LIMIT;
        return wraps ? static_cast<T>(new_unadjusted_val - LIMIT) : static_cast<T>(new_unadjusted_val);
    }
}

FORCE_INLINE
auto normalize_ptr(BufferPtr ptr, uint8_t num_buffers) -> BufferIndex {
    // note it may make sense to calculate this only when we increment
    // which will save calculations overall (but may add register pressure)
    // and introduce undesirable loads
    bool normalize = ptr >= num_buffers;
    uint8_t normalized_ptr = ptr.get() - static_cast<uint8_t>(normalize * num_buffers);
    ASSERT(normalized_ptr < num_buffers);
    return BufferIndex{normalized_ptr};
}
template <uint8_t NUM_BUFFERS>
FORCE_INLINE auto normalize_ptr(BufferPtr ptr) -> BufferIndex {
    static_assert(NUM_BUFFERS != 0, "normalize_ptr called with NUM_BUFFERS of 0; it must be greater than 0");
    constexpr bool is_size_pow2 = NUM_BUFFERS != 0 && (NUM_BUFFERS & (NUM_BUFFERS - 1)) == 0;
    constexpr bool is_size_2 = NUM_BUFFERS == 2;
    constexpr bool is_size_1 = NUM_BUFFERS == 1;
    constexpr uint8_t wrap_mask = NUM_BUFFERS - 1;
    if constexpr (is_size_pow2) {
        return BufferIndex{static_cast<uint8_t>(ptr.get() & wrap_mask)};
    } else if constexpr (is_size_2) {
        return BufferIndex{(uint8_t)1 - ptr.get()};
    } else if constexpr (is_size_1) {
        return BufferIndex{0};
    } else {
        // note it may make sense to calculate this only when we increment
        // which will save calculations overall (but may add register pressure)
        // and introduce undesirable loads
        return normalize_ptr(ptr, NUM_BUFFERS);
    }
}

FORCE_INLINE uint8_t
distance_behind(const BufferPtr& trailing_ptr, const BufferPtr& leading_ptr, uint8_t ptr_wrap_size) {
    bool leading_gte_trailing_ptr = leading_ptr >= trailing_ptr;
    return leading_gte_trailing_ptr ? leading_ptr - trailing_ptr : ptr_wrap_size - (trailing_ptr - leading_ptr);
}
template <uint8_t NUM_BUFFERS>
FORCE_INLINE uint8_t distance_behind(const BufferPtr& trailing_ptr, const BufferPtr& leading_ptr) {
    static_assert(NUM_BUFFERS != 0, "distance_behind called with NUM_BUFFERS of 0; it must be greater than 0");
    constexpr bool is_size_pow2 = is_power_of_2(NUM_BUFFERS);
    constexpr uint8_t ptr_wrap_mask = (2 * NUM_BUFFERS) - 1;
    constexpr uint8_t ptr_wrap_size = 2 * NUM_BUFFERS;
    if constexpr (is_size_pow2) {
        return (leading_ptr - trailing_ptr) & ptr_wrap_mask;
    } else {
        return distance_behind(trailing_ptr, leading_ptr, ptr_wrap_size);
    }
}

template <uint8_t NUM_BUFFERS>
class ChannelBufferPointer {
    static_assert(
        NUM_BUFFERS <= std::numeric_limits<uint8_t>::max() / 2,
        "NUM_BUFFERS must be less than or half of std::numeric_limits<uint8_t>::max() due to the internal "
        "implementation");

public:
    static constexpr bool is_size_pow2 = (NUM_BUFFERS & (NUM_BUFFERS - 1)) == 0;
    static constexpr bool is_size_2 = NUM_BUFFERS == 2;
    static constexpr bool is_size_1 = NUM_BUFFERS == 1;
    static constexpr uint8_t ptr_wrap_size = 2 * NUM_BUFFERS;

    // Only to use if is_size_pow2
    static constexpr uint8_t ptr_wrap_mask = (2 * NUM_BUFFERS) - 1;
    static constexpr uint8_t buffer_wrap_mask = NUM_BUFFERS - 1;
    ChannelBufferPointer() : ptr(0) {}
    /*
     * Returns the "raw" pointer - not usable to index the buffer channel
     */
    FORCE_INLINE BufferPtr get_ptr() const { return this->ptr; }

    FORCE_INLINE bool is_caught_up_to(const ChannelBufferPointer<NUM_BUFFERS>& leading_ptr) const {
        return this->is_caught_up_to(leading_ptr.get_ptr());
    }
    FORCE_INLINE uint8_t distance_behind(const ChannelBufferPointer<NUM_BUFFERS>& leading_ptr) const {
        return this->distance_behind(leading_ptr.get_ptr());
    }

    /*
     * Returns the buffer index pointer which is usable to index into the buffer memory
     */
    FORCE_INLINE BufferIndex get_buffer_index() const { return BufferIndex{normalize_ptr<NUM_BUFFERS>(this->ptr)}; }

    FORCE_INLINE void increment_n(uint8_t n) {
        this->ptr = BufferPtr{wrap_increment_n<2 * NUM_BUFFERS>(this->ptr.get(), n)};
    }
    FORCE_INLINE void increment() { this->ptr = BufferPtr{wrap_increment<2 * NUM_BUFFERS>(this->ptr.get())}; }

private:
    // Make these private to make sure caller doesn't accidentally mix two pointers pointing to
    // different sized channels
    FORCE_INLINE bool is_caught_up_to(const BufferPtr& leading_ptr) const { return this->get_ptr() == leading_ptr; }
    FORCE_INLINE uint8_t distance_behind(const BufferPtr& leading_ptr) const {
        return tt::tt_fabric::distance_behind<NUM_BUFFERS>(this->ptr, leading_ptr);
    }
    BufferPtr ptr = BufferPtr{0};
};

// Must call reset() before using
template <uint8_t NUM_BUFFERS>
struct ChannelCounter {
    static constexpr bool IS_POW2_NUM_BUFFERS = is_power_of_2(NUM_BUFFERS);
    uint32_t counter;
    BufferIndex index;

    FORCE_INLINE void reset() {
        this->counter = 0;
        this->index = BufferIndex(0);
    }

    FORCE_INLINE BufferIndex get_buffer_index() const { return index; }

    FORCE_INLINE void increment() {
        counter++;
        index = BufferIndex{wrap_increment<NUM_BUFFERS>(index.get())};
    }

    FORCE_INLINE void increment_n(uint32_t n) {
        counter += n;
        index = BufferIndex{wrap_increment_n<NUM_BUFFERS>(index.get(), n)};
    }

    FORCE_INLINE bool is_caught_up_to(const ChannelCounter& leading_counter) const {
        return this->counter == leading_counter.counter;
    }

    FORCE_INLINE uint32_t distance_behind(const ChannelCounter& leading_counter) const {
        return leading_counter.counter - this->counter;
    }
};

/*
 * Tracks receiver channel pointers (from sender side)
 */
template <uint8_t RECEIVER_NUM_BUFFERS>
struct OutboundReceiverChannelPointers {
    uint32_t slot_size_bytes;
    uint32_t remote_receiver_channel_address_base;
    uint32_t remote_receiver_channel_address_ptr;
    uint32_t remote_receiver_channel_address_last;
    uint32_t num_free_slots;

    FORCE_INLINE void init() {
        this->slot_size_bytes = 0U;
        this->remote_receiver_channel_address_base = 0U;
        this->remote_receiver_channel_address_ptr = 0U;
        this->remote_receiver_channel_address_last = 0U;
        this->num_free_slots = RECEIVER_NUM_BUFFERS;
    }

    FORCE_INLINE void init(uint32_t const remote_receiver_buffer_address, uint32_t const slot_size_bytes) {
        this->slot_size_bytes = slot_size_bytes;
        this->remote_receiver_channel_address_base = remote_receiver_buffer_address;
        this->remote_receiver_channel_address_ptr = remote_receiver_buffer_address;
        this->remote_receiver_channel_address_last = remote_receiver_buffer_address + ((RECEIVER_NUM_BUFFERS - 1U) * slot_size_bytes);
        this->num_free_slots = RECEIVER_NUM_BUFFERS;
    }

    FORCE_INLINE bool has_space_for_packet() const { return num_free_slots; }

    FORCE_INLINE void advance_remote_receiver_buffer_pointer() {
        bool const is_last_buffer = remote_receiver_channel_address_ptr == remote_receiver_channel_address_last;
        remote_receiver_channel_address_ptr += slot_size_bytes;
        if(is_last_buffer) {
            remote_receiver_channel_address_ptr = remote_receiver_channel_address_base;
        }
    }
};

/*
 * Tracks receiver channel pointers (from receiver side). Must call reset() before using.
 */
template <uint8_t RECEIVER_NUM_BUFFERS>
struct ReceiverChannelPointers {
    ChannelCounter<RECEIVER_NUM_BUFFERS> wr_sent_counter;
    ChannelCounter<RECEIVER_NUM_BUFFERS> wr_flush_counter;
    ChannelCounter<RECEIVER_NUM_BUFFERS> ack_counter;
    ChannelCounter<RECEIVER_NUM_BUFFERS> completion_counter;
    std::array<uint8_t, RECEIVER_NUM_BUFFERS> src_chan_ids;

    FORCE_INLINE void set_src_chan_id(BufferIndex buffer_index, uint8_t src_chan_id) {
        src_chan_ids[buffer_index.get()] = src_chan_id;
    }

    FORCE_INLINE uint8_t get_src_chan_id(BufferIndex buffer_index) const { return src_chan_ids[buffer_index.get()]; }

    FORCE_INLINE uint8_t get_src_chan_id() const { return src_chan_ids[0]; }

    FORCE_INLINE void init() { reset(); }

    FORCE_INLINE void reset() {
        wr_sent_counter.reset();
        wr_flush_counter.reset();
        ack_counter.reset();
        completion_counter.reset();
    }
};

// Forward‐declare the Impl primary template:
template <template <uint8_t> class ChannelType, auto& BufferSizes, typename Seq>
struct ChannelPointersTupleImpl;

// Provide the specialization that actually holds the tuple and `get<>`:
template <template <uint8_t> class ChannelType, auto& BufferSizes, size_t... Is>
struct ChannelPointersTupleImpl<ChannelType, BufferSizes, std::index_sequence<Is...>> {
    static constexpr size_t N = sizeof...(Is);
    std::tuple<ChannelType<BufferSizes[Is]>...> channel_ptrs;

    template <size_t I>
    constexpr auto& get() {
        return std::get<I>(channel_ptrs);
    }
};

// Simplify the "builder" so that make() returns the Impl<…> directly:
template <template <uint8_t> class ChannelType, auto& BufferSizes>
struct ChannelPointersTuple {
    static constexpr size_t N = std::size(BufferSizes);

    static constexpr auto make() {
        // call init() on each element and return it
        auto channel_ptrs = ChannelPointersTupleImpl<ChannelType, BufferSizes, std::make_index_sequence<N>>{};
        std::apply(
            [&](auto&... chans) { ((chans.init()), ...); },
            channel_ptrs.channel_ptrs);  // Apply to the actual tuple member
        return channel_ptrs;
    }
};

}  // namespace tt::tt_fabric
