// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_recip.h"
#include <limits>
#include "llk_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <ApproximationMode APPROX_MODE, int ITERATIONS, int RECIPROCAL_ITERATIONS>
inline void calculate_rsqrt() {
    _calculate_rsqrt_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, FAST_APPROX, legacy_compat>(ITERATIONS);
}

template <ApproximationMode APPROX_MODE>
inline void rsqrt_init() {
    vConstFloatPrgm0 = 1.442695f;  // ln2_recip
    vConstFloatPrgm1 = 2.0f;
}

}  // namespace sfpu
}  // namespace ckernel
