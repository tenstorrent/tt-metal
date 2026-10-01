// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Samuel Jett (sjettTT, sjett@tenstorrent.com)'s Qwen model at cd9a11771107ea2c27da3303a0556ff7343e4af5 uses the
// former legacy reciprocal by default. Keep that arithmetic local to this model:
// current upstream's reciprocal has different rounding and owns SFPU constants,
// LOADMACRO configuration and replay slots. The legacy iteration owns none of them,
// allowing the source's sqrt/reciprocal sequence to retain its sqrt constants.
// Derived verbatim (apart from names and formatting) from the pinned source's
// tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_rsqrt_compat.h.

#include <limits>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel::sfpu {
template <int max_iter = 3>
sfpi_inline sfpi::vFloat qwen38_reciprocal_scalar(const sfpi::vFloat in) {
    // Force sign to 1 (make number negative)
    sfpi::vFloat val = sfpi::setsgn(in, 1);

    val = setexp(val, 126);  // Set exponent to 126 to make the number in 0.5-1
    // Use 1.44 as first guess at x, ideal value would be 1.33.
    // Grayskull has hardwired 1.44 and uses it to avoid a load.
    // We use it here for consistency.
    sfpi::vFloat vConstLn2Recip = 1.442695f;
    // The pinned source compiler emits SFPMUL followed by SFPADDI, with a
    // rounding boundary between them. An ordinary vector addition now contracts
    // into SFPMAD, changing the real-checkpoint gated norm. Keep the immediate
    // addition explicit, local to the source-policy reciprocal.
    auto add_two = [](sfpi::vFloat product) -> sfpi::vFloat {
        return __builtin_rvtt_sfpaddi(ckernel::instrn_buffer, product.get(), 0x4000, 0, 0, 0);
    };
    sfpi::vFloat result = vConstLn2Recip * add_two(val * vConstLn2Recip);

    for (int s_iter = 0; s_iter < (max_iter - 1); s_iter++) {
        result = result * add_two(val * result);
    }

    sfpi::vInt orig_exp = exexp(in);
    sfpi::vInt new_exp = exexp(result);

    // "Subtract" exponents, and re-bias.
    // Execute: -1 - exp, then exp += 127
    new_exp -= orig_exp;
    new_exp += 126;

    v_if(new_exp < 0) {
        // If rebiased exponent is negative, we need to saturate at 0.
        // This means the initial number was too big so reciprocal result should be 0
        result = 0.0F;
        new_exp = 0;
    }
    v_endif;

    // Set newly denormalized exponent to result exponent field
    sfpi::vFloat out = sfpi::setexp(result, new_exp);

    // Pole guard for in == 0, which the exponent-difference arithmetic above misses: it lands
    // on 126 - exexp(0) = 254, a finite 1.7e38, where an infinity needs 255. The
    // v_if(new_exp < 0) block guards only the opposite, underflow end. Issue #52930.
    //
    // Two constraints on the form. It has to run after the setexp, which would otherwise
    // overwrite the exponent field that makes the value an infinity. And it has to compare
    // setsgn(in, 0) rather than a bare in == 0.0F, because SFPSETCC is not specified for
    // negative zero (VectorUnit.md) and leaves -0.0 at 1.7e38; clearing the sign is what
    // brings -0.0 into the guard, after which the caller-side v_if(in < 0.0) re-signs it.
    v_if(sfpi::setsgn(in, 0) == 0.0F) { out = std::numeric_limits<float>::infinity(); }
    v_endif;
    return out;
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en>
inline void qwen38_calculate_reciprocal(const int iterations) {
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat out = qwen38_reciprocal_scalar<APPROXIMATION_MODE ? 2 : 3>(in);
        v_if(in < 0.0) { out = -out; }
        v_endif;
        if constexpr (!(fp32_dest_acc_en || APPROXIMATION_MODE)) {
            out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = out;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void qwen38_reciprocal_init() {
    _init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}
}  // namespace ckernel::sfpu
#endif

namespace ckernel {
ALWI void qwen38_recip_tile_init() { MATH(SFPU_UNARY_INIT_FN(reciprocal, sfpu::qwen38_reciprocal_init, (APPROX))); }
ALWI void qwen38_recip_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, qwen38_calculate_reciprocal, (APPROX, 8, DST_ACCUM_MODE), idst, vector_mode, 8));
}
}  // namespace ckernel
