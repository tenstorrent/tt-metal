// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Frozen copies of the Blackhole FP32-accurate sigmoid / SiLU / exp kernels and their inits as they were before the
// SFPU cycle reductions (skrstic/kda-block-max, 23faf828355), in namespace ckernel::sfpu::ref. The bit-identity test
// runs each next to the production kernel on every input bit pattern and compares the results bit for bit. Only the
// helpers these copies share with production (ExpFp32Constants, the exp_21f coefficients and int conversion,
// sfpu_reciprocal_iter, PolynomialEvaluator, Converter) come from the production headers, which the reductions leave
// unchanged.

#pragma once

#include <limits>

#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sigmoid.h"

namespace ckernel::sfpu::ref {

struct SigmoidBf16Constants {
    sfpi::vFloat neg_one_ln2 = -1.4426950216293334961f;
    sfpi::vFloat exp_bias = 127.f;
    sfpi::vFloat c0 = EXP_21F_BF16_C0;
    sfpi::vFloat c1 = EXP_21F_BF16_C1;
    sfpi::vFloat c2 = EXP_21F_BF16_C2;
};

template <typename Constants>
sfpi_inline sfpi::vFloat sigmoid_bf16_preloaded(sfpi::vFloat x, const Constants& k) {
    sfpi::vFloat xlog2 = sfpi::clamp(x * k.neg_one_ln2 + k.exp_bias, 0.0f, 254.0f);
    sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));
    sfpi::vInt exponential_part = sfpi::exexp(z, sfpi::ExponentMode::Biased);
    sfpi::vMag fractional_part = sfpi::exman(z);
    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);
    frac = PolynomialEvaluator::eval(frac, k.c0, k.c1, k.c2);
    sfpi::vFloat exp_neg_x = sfpi::setexp(frac, exponential_part);
    return sfpu_reciprocal_iter<1, /*normalized=*/true>(1.0f + exp_neg_x);
}

struct SigmoidFp32Constants {
    sfpi::vFloat neg_one_ln2 = -ExpFp32Constants::one_ln2;
};

template <int N>
sfpi_inline void sigmoid_fp32_rows(sfpi::vFloat (&x)[N], const SigmoidFp32Constants& constants) {
    using C = ExpFp32Constants;
    sfpi::vFloat j[N], f[N], s[N], r[N];
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        j[n] = sfpi::abs(x[n]);
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        j[n] = sfpi::min(j[n], sfpi::vFloat(88.0f));
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        x[n] = sfpi::copysgn(j[n], x[n]);
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        j[n] = x[n] * constants.neg_one_ln2;
    }
    sfpi::vSMag16 sm[N];
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        sm[n] = sfpi::convert<sfpi::vSMag16>(j[n], sfpi::RoundMode::Nearest);
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        j[n] = sfpi::convert<sfpi::vFloat>(sm[n], sfpi::RoundMode::Nearest);
    }
    sfpi::vFloat k = C::neg_ln2_hi;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        f[n] = j[n] * k - x[n];
    }
    k = C::neg_ln2_lo;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        f[n] = j[n] * k + f[n];
    }
    k = Converter::as_float(0x4B400000U + 127U);
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        s[n] = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(j[n] + k) << 23);
    }
    k = C::c5;
    sfpi::vFloat c = C::c4;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = k * f[n] + c;
    }
    c = C::c3;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = r[n] * f[n] + c;
    }
    c = C::c2;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = r[n] * f[n] + c;
    }
    c = C::c1;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = r[n] * f[n] + c;
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = r[n] * f[n] + 1.0f;
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = r[n] * f[n] + 1.0f;
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        f[n] = r[n] * s[n] + 1.0f;
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        s[n] = sfpi::approx_recip(f[n]);
    }
#pragma GCC unroll 2
    for (int i = 0; i < 2; i++) {
#pragma GCC unroll 2
        for (int n = 0; n < N; n++) {
            r[n] = sfpi::vConstFloatPrgm0 - f[n] * s[n];
        }
#pragma GCC unroll 2
        for (int n = 0; n < N; n++) {
            s[n] = s[n] * r[n];
        }
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        x[n] = s[n];
    }
}

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sigmoid() {
    if constexpr (!is_fp32_dest_acc_en) {
        const SigmoidBf16Constants constants;
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat result = sigmoid_bf16_preloaded(sfpi::dst_reg[0], constants);
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    } else {
        const SigmoidFp32Constants constants;
#pragma GCC unroll 4
        for (int d = 0; d < ITERATIONS / 2; d++) {
            sfpi::vFloat rows[2] = {sfpi::dst_reg[0], sfpi::dst_reg[1]};
            sigmoid_fp32_rows(rows, constants);
            sfpi::dst_reg[0] = rows[0];
            sfpi::dst_reg[1] = rows[1];
            sfpi::dst_reg += 2;
        }
        if constexpr (ITERATIONS % 2 != 0) {
            sfpi::vFloat row[1] = {sfpi::dst_reg[0]};
            sigmoid_fp32_rows(row, constants);
            sfpi::dst_reg[0] = row[0];
            sfpi::dst_reg++;
        }
    }
}

inline void sigmoid_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<false>();
}

struct SiluBf16Constants {
    sfpi::vFloat neg_one_ln2 = -1.4426950216293334961f;
    sfpi::vFloat exp_bias = 127.f;
    sfpi::vFloat c0 = EXP_21F_BF16_C0;
    static constexpr auto& c1 = sfpi::vConstFloatPrgm1;
    static constexpr auto& c2 = sfpi::vConstFloatPrgm2;
};

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
    if constexpr (!is_fp32_dest_acc_en) {
        const SiluBf16Constants constants;
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];
            sfpi::vFloat result = x * sigmoid_bf16_preloaded(x, constants);
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
        return;
    }
    const SigmoidFp32Constants constants;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS / 2; d++) {
        sfpi::vFloat rows[2] = {sfpi::dst_reg[0], sfpi::dst_reg[1]};
        sigmoid_fp32_rows(rows, constants);
        sfpi::vFloat x0 = sfpi::dst_reg[0];
        sfpi::vFloat x1 = sfpi::dst_reg[1];
        sfpi::dst_reg[0] = x0 * rows[0];
        sfpi::dst_reg[1] = x1 * rows[1];
        sfpi::dst_reg += 2;
    }
    if constexpr (ITERATIONS % 2 != 0) {
        sfpi::vFloat row[1] = {sfpi::dst_reg[0]};
        sigmoid_fp32_rows(row, constants);
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * row[0];
        sfpi::dst_reg++;
    }
}

inline void silu_init() {
    sigmoid_init();
    sfpi::vConstFloatPrgm1 = EXP_21F_BF16_C1;
    sfpi::vConstFloatPrgm2 = EXP_21F_BF16_C2;
}

template <int N>
sfpi_inline void exp_fp32_accurate_rows(sfpi::vFloat (&a)[N]) {
    using C = ExpFp32Constants;
    sfpi::vFloat j[N], f[N], r[N];
    sfpi::vInt i[N];
    sfpi::vFloat k = C::one_ln2;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        j[n] = k * a[n];
    }
    sfpi::vSMag16 sm[N];
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        sm[n] = sfpi::convert<sfpi::vSMag16>(j[n], sfpi::RoundMode::Nearest);
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        j[n] = sfpi::convert<sfpi::vFloat>(sm[n], sfpi::RoundMode::Nearest);
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        i[n] = sfpi::abs(sfpi::as<sfpi::vInt>(sm[n]));
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        i[n] = sfpi::as<sfpi::vInt>(sfpi::copysgn(sfpi::as<sfpi::vFloat>(i[n]), j[n]));
    }
    k = C::neg_ln2_hi;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        f[n] = j[n] * k + a[n];
    }
    k = C::neg_ln2_lo;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        f[n] = j[n] * k + f[n];
    }
    k = C::c5;
    sfpi::vFloat c = C::c4;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = k * f[n] + c;
    }
    c = C::c3;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = r[n] * f[n] + c;
    }
    c = C::c2;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = r[n] * f[n] + c;
    }
    c = C::c1;
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = r[n] * f[n] + c;
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = r[n] * f[n] + 1.0f;
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        r[n] = r[n] * f[n] + 1.0f;
    }
#pragma GCC unroll 2
    for (int n = 0; n < N; n++) {
        sfpi::vFloat y = r[n] * std::numeric_limits<float>::infinity();
        sfpi::vInt e = sfpi::exexp(r[n], sfpi::ExponentMode::Biased) + i[n];
        v_if(sfpi::nearby(e < 255)) {
            y = sfpi::setexp(r[n], e);
            v_if(sfpi::nearby(e < 1)) { y = 0.0f; }
            v_endif;
        }
        v_endif;
        a[n] = y;
    }
}

// calculate_exponential<false, true /*FP32 dest*/, false, ITERATIONS>.
template <int ITERATIONS>
inline void calculate_exponential_fp32() {
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS / 2; d++) {
        sfpi::vFloat rows[2] = {sfpi::dst_reg[0], sfpi::dst_reg[1]};
        exp_fp32_accurate_rows(rows);
        sfpi::dst_reg[0] = rows[0];
        sfpi::dst_reg[1] = rows[1];
        sfpi::dst_reg += 2;
    }
    if constexpr (ITERATIONS % 2 != 0) {
        sfpi::vFloat row[1] = {sfpi::dst_reg[0]};
        exp_fp32_accurate_rows(row);
        sfpi::dst_reg[0] = row[0];
        sfpi::dst_reg++;
    }
}

// exp_init<false, ..., true /*FP32 dest*/>.
inline void exp_init_fp32() {
    sfpu::_init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<false>();
}

}  // namespace ckernel::sfpu::ref
