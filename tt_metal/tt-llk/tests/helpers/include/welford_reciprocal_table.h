// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

// Entry i is the fp32 bits of 1 / (i + 1), the reciprocal table the Welford kernels index by sample count.
template <std::size_t N>
inline void fill_welford_reciprocal_table(std::array<std::uint32_t, N>& table)
{
    for (std::uint32_t i = 0; i < N; ++i)
    {
        const float reciprocal = 1.0f / static_cast<float>(i + 1);
        std::uint32_t bits;
        __builtin_memcpy(&bits, &reciprocal, sizeof(bits));
        table[i] = bits;
    }
}
