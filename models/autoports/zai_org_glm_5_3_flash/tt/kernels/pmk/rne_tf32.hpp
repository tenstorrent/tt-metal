// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// (14's ttnn kda rne_tf32.hpp, tt-metal-s14rne b57d0652549, for the PMK KDA composite: GLM53_PMK_KDA_RNE. The switch
// here is the PMK_KDA_RNE define (the PMK kernels include this header only under it), not 14's math_approx_mode.)

// fp32 -> tf32 round-to-nearest-even, in place in DST (SFPU). The FPU reads fp32 CB tiles into srcA / srcB by
// TRUNCATING them to 19 bits (tf32: 10 mantissa bits), a -2^-11 mean magnitude bias per operand (GLM53 Stage 14
// round_probe2, 2026-10-25: matmul(a, I) == the tf32-truncation emulation on 100 % of elements). Rounding every fp32
// tile to tf32 RNE before it is packed makes the next FPU read exact: the precision stays tf32, the bias goes (and the
// mean squared error drops 4x: RNE vs truncation). Rounding a tile that is later read exactly (SFPU / unpack-to-dest)
// costs at most an unbiased tf32 RNE step.
// Usage (when on): kda_rne_tf32_init() once BEFORE the FPU op's own init (the FPU init then reprograms its address
// modes); kda_rne_tf32_tile(i) on each DST tile after the math, before tile_regs_commit().
#pragma once

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "llk_math_eltwise_unary_sfpu_macros.h"  // SFPU_UNARY_INIT / SFPU_UNARY_CALL_NO_TEMPLATE_ARGS (as exp.h)
#endif

#ifdef TRISC_MATH
#include "sfpi.h"

namespace ckernel::sfpu {

inline void kda_rne_tf32() {
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vUInt b = sfpi::as<sfpi::vUInt>(v);
        const sfpi::vUInt lsb = (b >> 13) & 1;  // the kept LSB: ties go to even
        b = b + 0x0FFF;
        b = b + lsb;
        b = (b >> 13) << 13;  // clear the 13 dropped mantissa bits (a carry rounds into the exponent)
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(b);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
#endif

#ifdef TRISC_MATH
inline void kda_rne_tf32_init_math() { SFPU_UNARY_INIT(unused); }
inline void kda_rne_tf32_tile_math(uint32_t idst) {
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, kda_rne_tf32, idst, VectorMode::RC);
}
#endif

// THE PACKER-GASKET MODE (PMK_KDA_RNE_PACK = GLM53_PMK_KDA_RNE=pack): the packer gasket rounds every DEST value it
// reads to a 10-bit mantissa (PCK_DEST_RD_CTRL_Round_10b_mant; HAL: "rounds to 10b mantissa, regardless of fp_pack
// format")
// -- the same tf32 words with no SFPU pass. pack_reconfig_data_format rewrites the bit (llk: on only for fp16 packs):
// arm after every reconfig to an fp32 CB, disarm at the body's end (the composite's next phase packs unrounded). The
// stall keeps an in-flight pack of the previous format out of the switch.
ALWI void kda_rne_pack_arm() {
    PACK((TTI_STALLWAIT(ckernel::p_stall::STALL_CFG, ckernel::p_stall::PACK)));
    PACK((ckernel::cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Round_10b_mant_RMW>(1)));
}
ALWI void kda_rne_pack_disarm() {
    PACK((TTI_STALLWAIT(ckernel::p_stall::STALL_CFG, ckernel::p_stall::PACK)));
    PACK((ckernel::cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Round_10b_mant_RMW>(0)));
}

ALWI void kda_rne_tf32_init() { MATH((kda_rne_tf32_init_math())); }

ALWI void kda_rne_tf32_tile(uint32_t idst) { MATH((kda_rne_tf32_tile_math(idst))); }
