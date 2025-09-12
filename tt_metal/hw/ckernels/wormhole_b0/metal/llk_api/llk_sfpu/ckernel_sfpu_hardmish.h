// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void hardmish() {
    // hardmish(x) = x * (x + 2.8).clamp(0.0, 5.0) / 5
    for (int d = 0; d < 8; d++) {
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
