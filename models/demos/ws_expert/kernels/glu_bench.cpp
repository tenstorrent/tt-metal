// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SFPU GLU microbench (PACK thread, fp32 DEST): gate / up tiles (CB 0, fp32, unpacked straight to DEST 0 / 1), the
// activation variant GLU_V applied N times into DEST 2 (variant 0 in place), then DEST 2 packed to CB 16.
// Stamps at the L1 address RT 0: [0] start, [1] end (after tensix_sync: the SFPU drained).
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/expert_activation.hpp"

#ifdef TRISC_PACK
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_sigmoid_appx.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
namespace ckernel::sfpu {
template <bool fp32, int ITERATIONS, int V>
inline void glu_v(const uint gi, const uint ui, const uint oi) {
    constexpr uint T = 32;
#if GLU_UNROLL == 2
#pragma GCC unroll 2
#else
#pragma GCC unroll 1
#endif
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat g = sfpi::dst_reg[gi * T];
        sfpi::vFloat u = sfpi::dst_reg[ui * T];
        sfpi::vFloat r;
        if constexpr (V == 1) {
            r = g * _sfpu_sigmoid_<true>(g) * u;
        } else if constexpr (V == 2) {
            r = g * _sfpu_sigmoid_<false>(g) * u;
        } else if constexpr (V == 3) {
            r = g * sfpi::approx_recip(1.0f + _sfpu_exp_21f_bf16_<true>(-g)) * u;
        } else if constexpr (V == 4) {
            r = g * u;
        } else if constexpr (V == 5) {
            r = g * u * _sfpu_exp_21f_bf16_<true>(-g);
        } else if constexpr (V == 6 || V == 7) {
            // t = exp(-|g|) in (0, 1]; s = 1 / (1 + t) in [0.5, 1) (normalized operand: one plain Newton step);
            // sigmoid(g) = 0.5 + sign(g) (s - 0.5)
            const sfpi::vFloat t = _sfpu_exp_21f_bf16_<true>(sfpi::setsgn(g, 1));
            const sfpi::vFloat den = t + 1.0f;
            sfpi::vFloat s = sfpi::approx_recip(den);
            if constexpr (V == 6) {
                s = s * (2.0f - den * s);
            }
            r = g * u * (sfpi::copysgn(s - 0.5f, g) + 0.5f);
        }
        sfpi::dst_reg[oi * T] = r;
        sfpi::dst_reg++;
    }
}

// v8: exp_21f with c0 / c1 / (1/ln2) in the programmable constant registers (no per-row SFPLOADI pairs for them),
// clamp to [0, 254] (finite: the Newton step never sees inf * 0), approx reciprocal + one Newton step
template <bool fp32, int ITERATIONS, int V>
inline void glu_v8(const uint gi, const uint ui, const uint oi) {
    constexpr uint T = 32;
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat g = sfpi::dst_reg[gi * T];
        const sfpi::vFloat gu = g * sfpi::dst_reg[ui * T];
        sfpi::vFloat xl = -g * sfpi::vConstFloatPrgm1 + 127.0f;
        xl = sfpi::clamp(xl, 0.0f, 254.0f);
        const sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xl));
        const sfpi::vInt ep = sfpi::exexp(z, sfpi::ExponentMode::Biased);
        sfpi::vFloat f = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
        f = (f * EXP_21F_C2 + sfpi::vConstFloatPrgm2) * f + sfpi::vConstFloatPrgm0;
        const sfpi::vFloat den = sfpi::setexp(f, ep) + 1.0f;
        sfpi::vFloat y = sfpi::approx_recip(den);
        y = y * (2.0f - den * y);
        sfpi::dst_reg[oi * T] = gu * y;
        sfpi::dst_reg++;
    }
}
inline void glu_v8_init() {
    sfpi::vConstFloatPrgm0 = EXP_21F_C0;
    sfpi::vConstFloatPrgm1 = EXP_21F_ONE_LN2;
    sfpi::vConstFloatPrgm2 = EXP_21F_C1;
}

// v9: the LUT sigmoid (sigmoid_appx: 3 linear pieces)
template <bool fp32, int ITERATIONS, int V>
inline void glu_v9(const uint gi, const uint ui, const uint oi) {
    constexpr uint T = 32;
    sfpi::vLut8si si0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vLut8si si1 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vLut8si si2 = sfpi::l_reg[sfpi::LRegs::LReg2];
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat g = sfpi::dst_reg[gi * T];
        sfpi::dst_reg[oi * T] = g * sfpi::dst_reg[ui * T] * (sfpi::lut(g, si0, si1, si2) + 0.5f);
        sfpi::dst_reg++;
    }
    sfpi::l_reg[sfpi::LRegs::LReg0] = si0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = si1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = si2;
}
}  // namespace ckernel::sfpu
#endif

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    constexpr uint32_t N = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(0, 16);
    copy_init(0);
#if GLU_V == 0
    expert_activation::init<expert_activation::Kind::Silu>();
#elif GLU_V == 8
    PACK((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::glu_v8_init)));
#elif GLU_V == 9
    PACK((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::sigmoid_appx_init)));
#else
    PACK((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::clamped_silu_glu_init)));
#endif
    cb_wait_front(0, 2);
    tile_regs_acquire();
    copy_tile(0, 0, 0);
    copy_tile(0, 1, 1);
    tile_regs_commit();
    cb_reserve_back(16, 1);
    tile_regs_wait();
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    PACK(tensix_sync());
    PACK(st[0] = wall());
    for (uint32_t i = 0; i < N; ++i) {
#if GLU_V == 0
        expert_activation::tile_pair<expert_activation::Kind::Silu>(0);
#elif GLU_V == 8
        PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, glu_v8, (DST_ACCUM_MODE, 8, 8), 0, 1, 2, VectorMode::RC)));
#elif GLU_V == 9
        PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, glu_v9, (DST_ACCUM_MODE, 8, 9), 0, 1, 2, VectorMode::RC)));
#else
        PACK((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, glu_v, (DST_ACCUM_MODE, 8, GLU_V), 0, 1, 2, VectorMode::RC)));
#endif
    }
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    PACK(tensix_sync());
    PACK(st[1] = wall());
    pack_tile(GLU_V == 0 ? 0 : 2, 16);
    tile_regs_release();
    cb_push_back(16, 1);
}
