// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <initializer_list>
#include <ostream>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <fmt/format.h>

namespace tt::tt_metal {

// A hardware counter with a specified number of bits
template <std::size_t BIT_COUNT = 17, typename UNDERLYING_TYPE = std::uint32_t>
class HardwareCounter {
private:
    static constexpr std::size_t kMask = (1ULL << BIT_COUNT) - 1;
    UNDERLYING_TYPE val_{0};

public:
    HardwareCounter() = default;

    // Create counter from an integer type
    template <typename IntegerType, typename std::enable_if_t<std::is_integral_v<IntegerType>, bool> = true>
    HardwareCounter(IntegerType other) {
        val_ = other & kMask;
    }

    HardwareCounter(const HardwareCounter& other) { val_ = other.val_; }

    HardwareCounter& operator=(const HardwareCounter& other) = default;

    HardwareCounter(const HardwareCounter&& other) = default;

    HardwareCounter& operator=(const HardwareCounter&& other) = default;

    // Enables casting to any integer type
    template <typename IntegerType, typename std::enable_if_t<std::is_integral_v<IntegerType>, bool> = true>
    operator IntegerType() const {
        return val_ & kMask;
    }

    // Returns the counter value
    UNDERLYING_TYPE value() const { return val_ & kMask; }

    // Returns the number of bits this counter supports
    std::size_t bit_count() const { return BIT_COUNT; }

    // Returns the max value of this counter
    std::size_t max() const { return 1ULL << BIT_COUNT; }

    // Reset the counter to zero
    void reset() { val_ = 0; }

    // Increment counter by specified value
    template <typename IntegerType, typename std::enable_if_t<std::is_integral_v<IntegerType>, bool> = true>
    HardwareCounter& operator+=(IntegerType value) {
        val_ += value;
        return *this;
    }

    HardwareCounter& operator++() {
        ++val_;
        return *this;
    }

    HardwareCounter operator++(int) {
        val_++;
        return *this;
    }

    // Decrement counter by specified value
    template <typename IntegerType, typename std::enable_if_t<std::is_integral_v<IntegerType>, bool> = true>
    HardwareCounter& operator-=(IntegerType value) {
        val_ -= value;
        return *this;
    }

    HardwareCounter& operator--() {
        --val_;
        return *this;
    }

    HardwareCounter operator--(int) {
        val_--;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const HardwareCounter& dt);
};

// std::cout support
template <std::size_t BIT_COUNT, typename UNDERLYING_TYPE>
std::ostream& operator<<(std::ostream& os, const HardwareCounter<BIT_COUNT, UNDERLYING_TYPE>& ct) {
    os << ct.val_;
    return os;
}

// fmt::format support
template <std::size_t BIT_COUNT, typename UNDERLYING_TYPE>
std::size_t format_as(const HardwareCounter<BIT_COUNT, UNDERLYING_TYPE>& ct) {
    return ct.value();
}

//
// NOC Auto Incrementing Stream Register
// NOTE: Number of bits hardcoded to number of stream register bits
//
using NOCAutoIncStreamReg = tt::tt_metal::HardwareCounter<17, std::uint32_t>;

}  // namespace tt::tt_metal
