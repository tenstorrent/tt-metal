// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "tt_metal/hw/inc/utils/utils.h"
#include "risc_attribs.h"

namespace tt::tt_fabric {

template <typename T, typename Parameter>
class NamedType {
public:
    FORCE_INLINE explicit NamedType(const T& value) : value_(value) {}
    FORCE_INLINE explicit NamedType(T&& value) : value_(std::move(value)) {}
    FORCE_INLINE NamedType<T, Parameter>& operator=(const NamedType<T, Parameter>& rhs) = default;
    FORCE_INLINE T& get() { return value_; }
    FORCE_INLINE const T& get() const { return value_; }
    FORCE_INLINE operator T() const { return value_; }
    FORCE_INLINE operator T&() { return value_; }

private:
    T value_;
};

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
    bool leading_gte_trailing_ptr = leading_ptr >= trailing_ptr;
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

}  // namespace tt::tt_fabric
