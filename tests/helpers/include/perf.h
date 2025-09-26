// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"

// FIXME: this shouldn't be statically allocated
constexpr uint32_t PERF_INPUT_A = 0x1A000;
constexpr uint32_t PERF_INPUT_B = PERF_INPUT_A + 16 * 4096;
constexpr uint32_t PERF_INPUT_C = PERF_INPUT_B + 16 * 4096;
constexpr uint32_t PERF_OUTPUT  = PERF_INPUT_C + 16 * 4096;

constexpr uint32_t PERF_ADDRESS(uint32_t buffer, uint32_t tile)
{
    uint32_t address = buffer + (tile % 16) * 4096; // Loop every 16 tiles, to prevent escaping memory
    return address / 16 - 1;                        // Correct the L1 Address for Tensix
}

enum class PerfRunType
{
    L1_TO_L1,
    UNPACK_ISOLATE,
    MATH_ISOLATE,
    PACK_ISOLATE,
    L1_CONGESTION
};

template <bool set_a, bool set_b>
inline void _perf_unpack_loop_set_valid(uint32_t iterations)
{
    while (iterations-- > 0)
    {
        constexpr uint32_t cond_clear_a = set_a ? ckernel::p_stall::SRCA_CLR : 0;
        constexpr uint32_t cond_clear_b = set_b ? ckernel::p_stall::SRCB_CLR : 0;
        TTI_SETDVALID((set_b << 1) | set_a);
#ifdef ARCH_QUASAR
        TTI_STALLWAIT(ckernel::p_stall::STALL_TDMA, cond_clear_a, cond_clear_b, 0);
#else
        TTI_STALLWAIT(ckernel::p_stall::STALL_TDMA, cond_clear_a | cond_clear_b);
#endif
    }
}

template <bool clear_a, bool clear_b>
inline void _perf_math_loop_clear_valid(uint32_t iterations)
{
    while (iterations-- > 0)
    {
        constexpr uint32_t cond_valid_a = clear_a ? ckernel::p_stall::SRCA_VLD : 0;
        constexpr uint32_t cond_valid_b = clear_b ? ckernel::p_stall::SRCB_VLD : 0;
#ifdef ARCH_QUASAR
        TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, cond_valid_a, cond_valid_b, 0);
        TTI_CLEARDVALID((clear_b << 1) | clear_a, 0, 0, 0, 0, 0);
#else
        TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, cond_valid_a | cond_valid_b);
        TTI_CLEARDVALID((clear_b << 1) | clear_a, 0);
#endif
    }
}
