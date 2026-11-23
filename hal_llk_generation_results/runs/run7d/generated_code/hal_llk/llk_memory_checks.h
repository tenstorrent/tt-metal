// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

/**
 * @brief Transform L1 address to the format used by LLK functions
 *
 * L1 address fields hold 16-byte units and are programmed one below the target block,
 * which the hardware increments before use: (address / 16) - 1.
 *
 * @param buffer_address The physical L1 address (byte-aligned)
 * @return Transformed address for LLK use: (address / 16) - 1
 */
constexpr inline std::uint32_t L1_ADDRESS(std::uint32_t buffer_address)
{
    return (buffer_address >> 4) - 1;
}
