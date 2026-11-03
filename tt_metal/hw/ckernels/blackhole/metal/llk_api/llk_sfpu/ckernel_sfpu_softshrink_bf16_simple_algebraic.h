// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace ckernel::sfpu::ttpoly::softshrink {
using namespace ::sfpi;
using ::sfpi::DataLayout;
template <bool NativeOrderedBranches = false>
sfpi_inline vFloat simple_threshold_softshift(vFloat x, float lambda) {
    if constexpr (NativeOrderedBranches) {
        // x - copysign(lambda, x) is x - lambda or x + lambda with the same single rounding; one compare of |x|
        // keeps it, so the row has one predicated stretch.
        vFloat shifted = x - copysgn(vFloat(lambda), x);
        vFloat y = 0.0f;
        v_if(setsgn(x, 0) > lambda) { y = shifted; }
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
}  // namespace ckernel::sfpu::ttpoly::softshrink
