// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_clamp.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void hardmish() {
    // hardmish(x) = x * (x + 2.8).clamp(0.0, 5.0) / 5
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0] + 2.8f;

        v_if(a < 0.0f) { a = 0.0f; }
        v_endif;

        v_if(a > 5.0f) { a = 5.0f; }
        v_endif;

        dst_reg[0] = dst_reg[0] * a * 0.2f;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
