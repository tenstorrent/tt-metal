// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"

enum class PerfRunType
{
    L1_TO_L1,
    UNPACK_ISOLATE,
    MATH_ISOLATE,
    PACK_ISOLATE,
    L1_CONGESTION
};

template <bool clear_a, bool clear_b>
inline void _perf_math_loop_clear_valid(uint32_t iterations)
{
    while (iterations-- > 0)
    {
        constexpr uint32_t cond_valid_a = clear_a ? ckernel::p_stall::SRCA_VLD : 0;
        constexpr uint32_t cond_valid_b = clear_b ? ckernel::p_stall::SRCB_VLD : 0;
        TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, cond_valid_a | cond_valid_b);
        TTI_CLEARDVALID((clear_b << 1) | clear_a, 0);
    }
}
