// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "sfpu/ckernel_sfpu_recip.h"

namespace ckernel {
namespace sfpu {

template <
    bool APPROXIMATION_MODE,
    bool EN_32BIT_DEST,
    int ITERATIONS = SFPU_ITERATIONS,
    [[maybe_unused]] bool legacy_compat = true>
inline void calculate_reciprocal() {
    static_assert(legacy_compat == true, "Non-default legacy_compat (false) not supported in Quasar reciprocal");
    // EN_32BIT_DEST (is_fp32_dest_acc_en) selects the full-precision Newton path (fp32 Dest,
    // non-approx) vs the HW LUT used for every bf16 case and any explicit approx request.
    _calculate_reciprocal_<APPROXIMATION_MODE, EN_32BIT_DEST, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, [[maybe_unused]] bool EN_32BIT_DEST, [[maybe_unused]] bool legacy_compat = true>
void recip_init() {
    static_assert(legacy_compat == true, "Non-default legacy_compat (false) not supported in Quasar reciprocal");
    llk_math_eltwise_unary_sfpu_init<SfpuType::reciprocal>();
    // Program ADDR_MOD_6 (Dest post-increment of SFP_ROWS) and, for the non-approximate path, the
    // Newton-Raphson constant the reciprocal walks Dest / refines with.
    _init_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
