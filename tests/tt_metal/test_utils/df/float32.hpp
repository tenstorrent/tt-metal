// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <iostream>

#include <tt-metalium/logger.hpp>

namespace tt::test_utils::df {

//! Custom type is supported as long as the custom type supports the following custom functions
//! static SIZEOF - indicates byte size of custom type
//! to_float() - get float value from custom type
//! to_packed() - get packed (into an integral type that is of the bitwidth specified by SIZEOF)
//! Constructor(float in) - constructor with a float as the initializer

class float32 {
private:
    uint32_t uint32_data = 0;

public:
    static constexpr size_t SIZEOF = 4;

    // create from float: no rounding, just truncate
    float32(float float_num) {
        static_assert(sizeof float_num == sizeof uint32_data, "Can only support 32bit fp");
        // just move upper 16 to lower 16 (truncate)
        uint32_data = (*reinterpret_cast<uint32_t*>(&float_num));
    }

    // store lower 16 as 16-bit uint
    float32(uint32_t new_uint32_data) { uint32_data = new_uint32_data; }

    float to_float() const {
        float v;
        memcpy(&v, &uint32_data, sizeof(float));
        return v;
    }
    uint32_t to_packed() const { return uint32_data; }
    bool operator==(float32 rhs) { return uint32_data == rhs.uint32_data; }
    bool operator!=(float32 rhs) { return uint32_data != rhs.uint32_data; }
};

inline std::ostream& operator<<(std::ostream& os, const float32& val) {
    os << val.to_packed();
    return os;
}
}  // namespace tt::test_utils::df
