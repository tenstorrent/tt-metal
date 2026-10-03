// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "sfpi.h"

// Include polynomial_horner.h before this transport helper.
namespace sfpi {
template <uint32_t ConstantBits>
__attribute__((always_inline)) inline void dense_negative_nan_constant(vUInt raw, vFloat& result) {
    vUInt delta = (raw ^ vUInt(0x80ffu)) & vUInt(0x80ffu);
    v_if(delta == 0u) {
        vUInt mantissa = raw & vUInt(0x7f00u);
        v_if(mantissa != 0u) { result = __builtin_bit_cast(float, ConstantBits); }
        v_endif;
    }
    v_endif;
}
}  // namespace sfpi
