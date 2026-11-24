// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h, ported except SFPU. Same Tensix instruction sequence as production.

#pragma once
#include <cstdint>
#include <type_traits>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_sfpu_types.h"

using namespace ckernel;

// local function declarations
template <SfpuType sfpu_op>
inline void eltwise_unary_sfpu_configure_addrmod()
{
    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which is using ADDR_MOD_0 and ADDR_MOD_2, so use one
    //       that doesn't conflict!

    hal_port::set_sfpu_addr_mod<hal::cfg::Sec::S7 /*ADDR_MOD_7*/, 0>();

    if constexpr (sfpu_op == SfpuType::topk_local_sort)
    {
        hal_port::set_sfpu_addr_mod<hal::cfg::Sec::S6 /*ADDR_MOD_6*/, 32>();
    }

    if constexpr (
        sfpu_op == SfpuType::reciprocal || sfpu_op == SfpuType::typecast || sfpu_op == SfpuType::unary_max || sfpu_op == SfpuType::unary_min ||
        sfpu_op == SfpuType::unary_max_int32 || sfpu_op == SfpuType::unary_min_int32 || sfpu_op == SfpuType::unary_max_uint32 ||
        sfpu_op == SfpuType::unary_min_uint32 || sfpu_op == SfpuType::signbit || sfpu_op == SfpuType::equal_zero || sfpu_op == SfpuType::not_equal_zero ||
        sfpu_op == SfpuType::less_than_zero || sfpu_op == SfpuType::greater_than_equal_zero || sfpu_op == SfpuType::less_than_equal_zero ||
        sfpu_op == SfpuType::greater_than_zero)
    {
        hal_port::set_sfpu_addr_mod<hal::cfg::Sec::S6 /*ADDR_MOD_6*/, 2>();
    }
}

template <SfpuType sfpu_op>
inline void _llk_math_eltwise_unary_sfpu_init_()
{
    sfpu::_init_sfpu_config_reg();
    eltwise_unary_sfpu_configure_addrmod<sfpu_op>();
    hal_port::reset_counters_abd_f();
}

// Kernel-invariant SFPU init. Runs the parts of the per-op init that are identical for every SFPU op and
// therefore only need to run once per kernel: the SFPU config register (SFPCONFIG(0, 0xF, 1)) and the
// invariant ADDR_MOD_7 = {srca:0, srcb:0, dest:0}. Hoisted out of the per-op path so that the self-contained
// per-op init (ckernel::sfpu::_init_<op>_) does not re-run these on every op init. In Metal each per-op
// llk_math_eltwise_unary_sfpu_init runs this itself (#50381); the tt-llk standalone SFPU test harness runs
// it in its init prelude.
inline void _llk_math_eltwise_unary_sfpu_init_once_()
{
    sfpu::_init_sfpu_config_reg();

    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which is using ADDR_MOD_0 and ADDR_MOD_2, so use one
    //       that doesn't conflict!
    hal_port::set_sfpu_addr_mod<hal::cfg::Sec::S7 /*ADDR_MOD_7*/, 0>();
}
