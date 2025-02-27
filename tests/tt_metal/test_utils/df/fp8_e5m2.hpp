// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

class fp8_e5m2 {
private:
    uint8_t uint8_data = 0;

public:
    static constexpr size_t SIZEOF = 1;

    fp8_e5m2() = default;

    fp8_e5m2(float float_num) {
        static_assert(sizeof float_num == sizeof(uint32_t), "Can only support 32bit fp");
        uint32_t uint32_data = (*reinterpret_cast<uint32_t*>(&float_num));

        uint8_t sign = uint32_data >> 31;

        uint8_t exp = uint32_data >> 23;
        int16_t exp_unbias = exp - 127;

        uint32_t man = uint32_data & 0x7FFFFF;
        bool man_zero = man == 0;

        man >>= 21;

        if (exp == 0 || exp_unbias <= -15) {  // Flush to zero
            exp = 0;
            man = 0;
        } else if (exp == 0xFF || exp_unbias >= 16) {  // Flush to inf, but preserve NaNs
            exp = 0x1F;
            man = (exp == 0xFF && !man_zero) ? 1 : 0;
        } else {
            exp = exp_unbias + 15;
        }

        uint8_data = sign << 7;
        uint8_data |= exp << 2;
        uint8_data |= man;
    }

    fp8_e5m2(uint32_t new_uint8_data) { uint8_data = new_uint8_data; }

    float to_float() const {
        uint8_t sign = uint8_data >> 7;

        uint8_t exp = (uint8_data >> 2) & 0x1F;
        // If special exponent value, preserve, otherwise rebias
        if (exp == 0) {
            exp = 0;
        } else if (exp == 0x1F) {
            exp = 0xFF;
        } else {
            exp = exp + (127 - 15);
        }

        uint32_t man = uint8_data & 0x3;
        // Shift up mantissa to maintain the weight of high order bits
        man <<= 21;

        uint32_t uint32_data = sign << 31;
        uint32_data |= exp << 23;
        uint32_data |= man;

        float v;
        memcpy(&v, &uint32_data, sizeof(float));
        return v;
    }
    uint8_t to_packed() const { return uint8_data; }
    bool operator==(fp8_e5m2 rhs) { return uint8_data == rhs.uint8_data; }
    bool operator!=(fp8_e5m2 rhs) { return uint8_data != rhs.uint8_data; }
};

inline std::ostream& operator<<(std::ostream& os, const fp8_e5m2& val) {
    os << val.to_packed();
    return os;
}
}  // namespace tt::test_utils::df
