// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/uint8.hpp>

uint32_t pack_four_uint8_into_uint32(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    return static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 8) | (static_cast<uint32_t>(c) << 16) |
           (static_cast<uint32_t>(d) << 24);
}
