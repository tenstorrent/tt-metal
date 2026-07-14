// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::test_utils {

// Convert a sign-magnitude int8 byte (bit7=sign, bits[6:0]=magnitude) to a signed int.
inline int sign_mag_byte_to_int8(uint8_t byte) {
    const int mag = static_cast<int>(byte & 0x7F);
    const bool neg = (byte & 0x80) != 0;
    return neg ? -mag : mag;
}

// Convert a signed int32 to a sign-magnitude word (bit31=sign, bits[30:0]=magnitude).
inline uint32_t int32_to_sign_mag_word(int32_t value) {
    if (value >= 0) {
        return static_cast<uint32_t>(value);
    }
    return 0x80000000u | static_cast<uint32_t>(-static_cast<int64_t>(value));
}

}  // namespace tt::test_utils
