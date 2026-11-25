// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace ckernel::sfpu::ttpoly::erfc {
using ::sfpi::DataLayout;
using ::sfpi::vFloat;
using ::sfpi::vUInt;
__attribute__((always_inline)) inline vFloat target_raw_terminal_value(
    [[maybe_unused]] vFloat computed, float constant = 0.0f) {
    return vFloat(constant);
}

__attribute__((always_inline)) inline void signed_nonfinite_split_terminal(
    vUInt raw_u16, vFloat& result, float constant) {
    // One exponent-FF test serves the positive constant and the negative-NaN class (a nonzero negative
    // mantissa); negative infinity is left to the numeric body.
    vUInt exponent = raw_u16 & vUInt(0x00ffu);
    v_if(exponent == vUInt(0x00ffu)) {
        vUInt sign = raw_u16 & vUInt(0x8000u);
        v_if(sign == 0u) { result = target_raw_terminal_value(result, constant); }
        v_else {
            vUInt mantissa = raw_u16 & vUInt(0x7f00u);
            v_if(mantissa != 0u) { result = std::numeric_limits<float>::infinity(); }
            v_endif;
        }
        v_endif;
    }
    v_endif;
}
}  // namespace ckernel::sfpu::ttpoly::erfc
