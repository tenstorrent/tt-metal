// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <iostream>

namespace tt::test_utils::df {

//! Custom type is supported as long as the custom type supports the following custom functions
//! static SIZEOF - indicates byte size of custom type
//! to_float() - get float value from custom type
//! to_packed() - get packed (into an integral type that is of the bitwidth specified by SIZEOF)
//! Constructor(float in) - constructor with a float as the initializer

class bfloat16 {
   private:
    uint16_t uint16_data;

   public:
    static constexpr size_t SIZEOF = 2;

    constexpr bfloat16() noexcept = default;

    // create from float: no rounding, just truncate
    constexpr bfloat16(float float_num) noexcept : bfloat16((std::bit_cast<uint32_t>(float_num)) >> 16) {
        static_assert(sizeof(float) == sizeof(uint32_t), "Can only support 32bit fp");
    }

    // store lower 16 as 16-bit uint
    constexpr bfloat16(uint32_t uint32_data) noexcept : uint16_data(static_cast<uint16_t>(uint32_data)) {}

    constexpr float to_float() const noexcept { return std::bit_cast<float>(static_cast<uint32_t>(uint16_data) << 16); }

    constexpr uint16_t to_packed() const noexcept { return uint16_data; }

    constexpr bool operator==(const bfloat16& rhs) const noexcept = default;
};

inline std::ostream& operator<<(std::ostream& os, const bfloat16& val) {
    os << val.to_packed();
    return os;
}

}  // namespace tt::test_utils::df
