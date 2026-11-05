// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel_sfpu_mish_bf16_replay_window.h"
// Selected product bodies. Caller owns initialization and SFPU start/done.
namespace ckernel::sfpu::ttpoly::mish {
using namespace ::sfpi;
using ::sfpi::DataLayout;
template <class Config, class Pin>
__attribute__((always_inline)) inline vFloat product_bounded_wh_core(
    vFloat x, vFloat multiplier, vFloat c0, vFloat top, Pin pin_square) {
    vFloat x_original = x;
    vFloat bias_register = Config::kBias;
    vFloat xlog2 = __builtin_rvtt_sfpmad(x.get(), multiplier.get(), bias_register.get(), SFPMAD_MOD1_OFFSET_NONE);
    vInt e = exexp(xlog2);
    vInt m = exman(xlog2, MantissaMode::ImplicitOne);
    m = shft(m, e, ShiftMode::Logical);
    vFloat z = as<vFloat>(m);
    vFloat promoted_middle = Config::kCoefficients[1] * 0x1p-23f;
    vInt ep = exexp(z, ExponentMode::Biased);
    vMag fm = exman(z);
    vFloat f = convert<vFloat>(fm, RoundMode::Nearest);
    vFloat p = top * f + promoted_middle;
    p = p * f + c0;
    vFloat y = setexp(p, ep);
    vFloat shifted = y + Config::kScale;
    vFloat square_constant = pin_square(Config::kScale * Config::kScale);
    vFloat denominator = shifted * shifted + square_constant;
    vFloat negative_x = copyman(-1.0f, denominator);
    vFloat inv = vConstFloatPrgm1 + vConstFloatPrgm0 * negative_x;
    vUInt scale_bits = ~as<vUInt>(denominator);
    inv = vConstFloatPrgm2 + inv * negative_x;
    vFloat scale = setman(as<vFloat>(scale_bits), 0);
    vFloat t = 1.0f + negative_x * inv;
    scale *= 0.5f;
    inv = inv + inv * t;
    inv = inv * scale;
    vFloat negative_slope = -Config::kNegativeSlope;
    vFloat bias = -Config::kNegativeBias;
    vFloat negative_numerator =
        __builtin_rvtt_sfpmad(negative_slope.get(), y.get(), bias.get(), SFPMAD_MOD1_OFFSET_NONE);
    vFloat positive_scale = Config::kPositiveScale;
    vFloat negative_scale = positive_scale;
    vFloat one = 1.0f;
    vFloat factor = __builtin_rvtt_sfpmad(negative_scale.get(), inv.get(), one.get(), SFPMAD_MOD1_OFFSET_NONE);
    vFloat coordinate = x_original;
    v_if(x_original < 0.0f) {
        factor = negative_numerator * inv;
        coordinate = coordinate * y;
    }
    v_endif;
    y = coordinate * factor;
    return y;
}

template <class Config>
__attribute__((always_inline)) inline vFloat product_bounded_wh_prepare(vFloat raw) {
    vFloat prepared = raw;
    v_if(prepared < Config::kLowerBound) { prepared = Config::kLowerBound; }
    v_elseif(prepared > Config::kUpperBound) { prepared = Config::kUpperBound; }
    v_endif;
    return prepared;
}

template <class Config>
__attribute__((always_inline)) inline void product_bounded_wh_terminal(vFloat raw, vFloat& result) {
    constexpr uint32_t origin_bf16 = (Config::kOriginEncoded & 0x8000u) | ((Config::kOriginEncoded & 0xffu) << 7) |
                                     ((Config::kOriginEncoded >> 8) & 0x7fu);
    v_if(as<vUInt>(raw) == vUInt(origin_bf16 << 16)) { result = -std::numeric_limits<float>::min(); }
    v_endif;
    vFloat minus_one = -1.0f;
    vFloat upper = Config::kDomainActions[2].bound;
    vFloat upper_distance = __builtin_rvtt_sfpmad(minus_one.get(), raw.get(), upper.get(), SFPMAD_MOD1_OFFSET_NONE);
    vFloat lower = Config::kDomainActions[1].bound;
    vFloat lower_distance = __builtin_rvtt_sfpmad(minus_one.get(), raw.get(), lower.get(), SFPMAD_MOD1_OFFSET_NONE);
    v_if(upper_distance < 0.0f) { result = raw; }
    v_endif;
    v_if(lower_distance >= 0.0f) { result = raw * 0.0f; }
    v_endif;
}

template <LRegs Register>
sfpi_inline vFloat product_pin(vFloat value) {
    l_reg[Register] = value;
    return l_reg[Register];
}
template <class Config>
inline void product_bounded_wh_tile() {
    vFloat top = product_pin<LRegs::LReg5>(Config::kCoefficients[2] * 0x1p-46f);
    vFloat c0 = product_pin<LRegs::LReg7>(Config::kCoefficients[0]);
    vFloat multiplier = product_pin<LRegs::LReg6>(Config::kMultiplier);
#pragma GCC unroll 32
    for (int row = 0; row < 32; ++row) {
        vFloat x = product_bounded_wh_prepare<Config>(dst_reg[row]);
        vFloat result =
            product_bounded_wh_core<Config>(x, multiplier, c0, top, [](vFloat value) __attribute__((always_inline)) {
                return product_pin<LRegs::LReg4>(value);
            });
        vFloat raw = dst_reg[row];
        product_bounded_wh_terminal<Config>(raw, result);
        result = convert<vFloat16b>(result, RoundMode::Nearest);
        dst_reg[row] = result;
    }
}

}  // namespace ckernel::sfpu::ttpoly::mish
