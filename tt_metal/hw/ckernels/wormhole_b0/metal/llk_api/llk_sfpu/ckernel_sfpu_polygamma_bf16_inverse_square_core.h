// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace ckernel::sfpu::ttpoly::polygamma {
using namespace ::sfpi;
using ::sfpi::DataLayout;
template <class Config, class Reciprocal, class Tail>
__attribute__((always_inline)) inline vFloat inverse_square_scalar(
    vFloat x, Reciprocal reciprocal, [[maybe_unused]] Tail tail) {
    vFloat zero = 0.0f;
    vFloat one = 1.0f;
    // psi1(x) = psi1(x+1) + 1/x^2: one unconditional step lets both arms share z = 1 + |x|.
    vFloat z = setsgn(x, 0) + one;
    constexpr float kRound = 0x1.8p23f;
    vFloat n = x + kRound;
    n = n - kRound;
    vFloat signed_one = copysgn(one, x);
    vFloat negative_mask = 0.5f - 0.5f * signed_one;
    vFloat q = x - negative_mask * n;
    vFloat u = reciprocal(z);
    // The Bernoulli core in Horner order.
    vFloat core = ((Config::kP0 * u + 0.5f) * u + one) * u;
    vFloat iq = reciprocal(q);
    // Mask before squaring: the dead regularizer would overflow at large x and turn 0 * it into NaN.
    vFloat regularizer_q = negative_mask * q;
    vFloat q2 = regularizer_q * regularizer_q;
    vFloat regularizer = (Config::kP2[2] * q2 + Config::kP2[1]) * q2 + Config::kP2[0];
    vFloat result = iq * iq + copysgn(core, x);
    result = negative_mask * regularizer + result;
    v_if((x < zero) && (q == zero)) { result = std::numeric_limits<float>::infinity(); }
    v_endif;
    vInt source_exponent = exexp(setsgn(x, 0), ExponentMode::Biased);
    v_if(source_exponent == 255) { result = zero; }
    v_endif;
    v_if(is_zero(x)) { result = std::numeric_limits<float>::infinity(); }
    v_endif;
    v_if(x == 0x1p126f) { result = 0x1p-126f; }
    v_endif;
    return result;
}
}  // namespace ckernel::sfpu::ttpoly::polygamma
