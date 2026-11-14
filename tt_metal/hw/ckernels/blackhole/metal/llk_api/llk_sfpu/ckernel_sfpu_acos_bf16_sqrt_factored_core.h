// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace ckernel::sfpu::ttpoly::acos {
using namespace ::sfpi;
using ::sfpi::DataLayout;
__attribute__((always_inline)) inline vFloat basis_sqrt(vFloat x) {
    vInt i = as<vInt>(as<vUInt>(x) >> 1);
    vFloat y = as<vFloat>(0x5f1110a0 - i);
    vFloat xy = x * y;
    vFloat c = -y * xy;
    y = y * (2.2825186f + c * (2.2533049f + c));
    xy = x * y;
    vFloat one_minus_xyy = vFloat(1.0f) + (-y * xy);
    vFloat half_xy = xy * 0.5f;
    y = one_minus_xyy * half_xy + xy;
    return y;
}

__attribute__((always_inline)) inline vFloat sqrt_factored_product(vFloat polynomial, vFloat magnitude) {
    return polynomial * basis_sqrt(vFloat(1.0f) - magnitude);
}

__attribute__((always_inline)) inline vFloat sqrt_factored_reflect(vFloat input, vFloat result) {
    constexpr float pi = 3.14159265358979323846f;
    v_if(input < 0.0f) { result = pi - result; }
    v_endif;
    return result;
}
}  // namespace ckernel::sfpu::ttpoly::acos
