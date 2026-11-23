// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

/**
 * @brief Transform an L1 byte address to the 16-byte, minus-one form used by Tensix L1 address fields.
 */
constexpr inline std::uint32_t L1_ADDRESS(std::uint32_t buffer_address)
{
    return (buffer_address >> 4) - 1;
}
