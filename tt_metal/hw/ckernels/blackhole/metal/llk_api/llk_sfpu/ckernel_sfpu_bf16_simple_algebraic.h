// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel_sfpu_bf16_sfpi_isa.h"
// Include inside namespace sfpi after sfpi_min_max.h; callers own traversal/init.

template <bool NativeOrderedBranches = false>
sfpi_inline vFloat simple_threshold_softshift(vFloat x, float lambda) {
    if constexpr (NativeOrderedBranches) {
        vFloat y = 0.0f;
        v_if(x > lambda) { y = x - lambda; }
        v_elseif(x < -lambda) { y = x + lambda; }
        v_endif;
        return y;
    } else {
        vFloat ax = setsgn(x, 0);
        vFloat y = 0.0f;
        v_if(ax > lambda) {
            y = ax - lambda;
            v_if(x < 0.0f) { y = -y; }
            v_endif;
        }
        v_endif;
        return y;
    }
}
