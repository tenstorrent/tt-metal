// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "llk_defs.h"

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_recip.h"
using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATE = false, bool save_reg = true /* Unused. Enough registers available. */>
sfpi_inline vFloat sfpu_reciprocal(const vFloat in) {
    return _sfpu_reciprocal_<APPROXIMATE ? 0 : 2>(in);
}

template <ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_reciprocal() {
    _calculate_reciprocal_<(APPROX_MODE == ApproximationMode::Fast), ITERATIONS, is_fp32_dest_acc_en>(ITERATIONS);
}

template <ApproximationMode APPROX_MODE>
void recip_init() {
    _init_reciprocal_<(APPROX_MODE == ApproximationMode::Fast)>();
}

}  // namespace sfpu
}  // namespace ckernel
