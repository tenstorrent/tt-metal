// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <iostream>

#include "tt_metal/common/logger.hpp"

using namespace std;

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

    bfloat16() : uint16_data(0) {}

    // create from float: no rounding, just truncate
    bfloat16(float float_num) {
        uint32_t uint32_data;
        static_assert(sizeof(float) == sizeof(uint32_t), "Can only support 32bit fp");
        uint32_data = *reinterpret_cast<uint32_t*>(&float_num);
        // just move upper 16 to lower 16 (truncate)
        uint32_data = (uint32_data >> 16);
        // store lower 16 as 16-bit uint
        uint16_data = (uint16_t)uint32_data;
    }

    // store lower 16 as 16-bit uint
    bfloat16(uint32_t uint32_data) { uint16_data = (uint16_t)uint32_data; }

    float to_float() const {
        uint32_t uint32_data = ((uint32_t)uint16_data) << 16;
        float f;
        std::memcpy(&f, &uint32_data, sizeof(f));
        return f;
    }
    uint16_t to_packed() const { return uint16_data; }
    bool operator==(const bfloat16 rhs) const { return uint16_data == rhs.uint16_data; }
    bool operator!=(const bfloat16 rhs) const { return uint16_data != rhs.uint16_data; }
};

inline ostream& operator<<(ostream& os, const bfloat16& val) {
    os << val.to_packed();
    return os;
}
}  // namespace tt::test_utils::df
