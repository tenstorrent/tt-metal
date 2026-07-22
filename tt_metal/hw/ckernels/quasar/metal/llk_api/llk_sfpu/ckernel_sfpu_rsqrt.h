// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "sfpu/ckernel_sfpu_rsqrt.h"

namespace ckernel {
namespace sfpu {

// Template-argument order mirrors the Blackhole/Wormhole calculate_rsqrt so the shared
// hw/inc/api compute layer can call it without an arch fork. FAST_APPROX / legacy_compat exist
// for ABI compatibility only and must stay at their defaults on Quasar.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    bool EN_32BIT_DEST = false,
    [[maybe_unused]] bool FAST_APPROX = false,
    [[maybe_unused]] bool legacy_compat = false>
inline void calculate_rsqrt() {
    static_assert(!FAST_APPROX, "Non-default FAST_APPROX (true) not supported in Quasar rsqrt");
    static_assert(!legacy_compat, "Non-default legacy_compat (true) not supported in Quasar rsqrt");
    // Two implementations only: APPROXIMATION_MODE (or a bf16 Dest) = HW LUT
    // (approx_recip(approx_sqrt)), else = full-precision fp32. EN_32BIT_DEST
    // (is_fp32_dest_acc_en) selects the full-precision path.
    _calculate_rsqrt_<APPROXIMATION_MODE, EN_32BIT_DEST, ITERATIONS>();
}

// Signature mirrors Blackhole/Wormhole rsqrt_init (<APPROXIMATION_MODE, legacy_compat>); the init
// itself does not depend on the Dest width, so no fp32 template arg is threaded here.
template <bool APPROXIMATION_MODE, [[maybe_unused]] bool legacy_compat = false>
void rsqrt_init() {
    static_assert(!legacy_compat, "Non-default legacy_compat (true) not supported in Quasar rsqrt");
    llk_math_eltwise_unary_sfpu_init<SfpuType::rsqrt>();
    // Program the SQRT_23-bits seed / refinement constants the full-precision rsqrt reads.
    _init_rsqrt_<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
