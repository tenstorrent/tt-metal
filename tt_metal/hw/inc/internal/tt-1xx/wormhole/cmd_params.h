// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <array>
#include <cassert>
#include <limits>
#include <cstring>
#ifndef DISABLE_CMD_DEBUG
#include <iostream>
#endif

#include "tensix.h"
#include "tensix_types.h"

// [[deprecated("There should be no more traditional fifos.")]]
inline std::uint32_t unpack_fifo_address(std::uint32_t fifo_address) {
    return (fifo_address << FIFO_BASE_ADDRESS_ALIGN_BITS);
}

inline std::uint32_t unpack_address(std::uint32_t address) { return (address << FIFO_BASE_ADDRESS_ALIGN_BITS); }

inline std::uint16_t pack_address(std::uint32_t address) {
#ifdef ASSERT
    ASSERT(
        !(address & bitmask<std::uint32_t>(FIFO_BASE_ADDRESS_ALIGN_BITS)), "Address not aligned and cannot be packed");
#else
    assert(
        !(address & bitmask<std::uint32_t>(FIFO_BASE_ADDRESS_ALIGN_BITS)) &&
        "Address not aligned and cannot be packed");
#endif
    return (address >> FIFO_BASE_ADDRESS_ALIGN_BITS);
}

inline std::uint32_t pack_32b_field(std::uint32_t x, unsigned int bits, unsigned int to_shift) {
    assert(bits + to_shift <= std::numeric_limits<std::uint32_t>::digits);
    assert((x & ~bitmask<std::uint32_t>(bits)) == 0);

    return x << to_shift;
}

inline std::uint32_t unpack_field(std::uint32_t x, unsigned int bits, unsigned int to_shift) {
    return ((x >> to_shift) & bitmask<std::uint32_t>(bits));
}

constexpr int MAX_NUM_PACKS = 4;
