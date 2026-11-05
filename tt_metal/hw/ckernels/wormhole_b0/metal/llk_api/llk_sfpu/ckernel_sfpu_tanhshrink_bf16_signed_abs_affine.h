// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Included after polynomial_horner.h and the target's existing LLK headers.
// Config carries the selected coefficient/terminal words. Callers own traversal.
namespace ckernel::sfpu::ttpoly::tanhshrink {
using namespace ::sfpi;
using ::sfpi::DataLayout;
template <typename Config>
__attribute__((always_inline)) inline void signed_abs_affine_tail(vFloat magnitude, vFloat& result) {
    v_if(magnitude > vFloat(__builtin_bit_cast(float, Config::kBoundBits))) {
        result = vFloat(__builtin_bit_cast(float, Config::kScaleBits)) * magnitude +
                 vFloat(__builtin_bit_cast(float, Config::kBiasBits));
    }
    v_endif;
}

}  // namespace ckernel::sfpu::ttpoly::tanhshrink
