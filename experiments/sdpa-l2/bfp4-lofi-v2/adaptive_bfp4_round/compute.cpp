// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// Fixed LREG0..7 only. DST tile0: original/result; tile1: current best
// normalized magnitudes; tile2: score at parity0 and power-of-two scale at
// parity1. Three FP32 DST tiles fit the four-tile half. No L1 scratch required.
inline void adaptive_max_2_3() {
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, 1);
    TTI_SFPNOP;
}

template <int ROTATIONS>
inline void adaptive_rotate_2_3() {
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG3, 3);
    TTI_SFPNOP;
    for (int i = 1; i < ROTATIONS; ++i) {
        TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG3, 3);
        TTI_SFPNOP;
    }
}

inline void adaptive_max_broadcast() {
    adaptive_rotate_2_3<1>();
    adaptive_max_2_3();
    adaptive_rotate_2_3<2>();
    adaptive_max_2_3();
    adaptive_rotate_2_3<4>();
    adaptive_max_2_3();
}

inline void adaptive_mse_reduce() {
    // R2 initially holds squared-even + squared-odd in each lane. Three
    // cyclic FP32 additions produce eight differently ordered sums. Select
    // only lane0 of each face row, then broadcast that one result; never let
    // per-lane rounding differences select inconsistent shared exponents.
    adaptive_rotate_2_3<1>();
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG2, 0);
    TTI_SFPNOP;  // MAD -> SHFT2 hazard is not automatically detected.
    adaptive_rotate_2_3<2>();
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    adaptive_rotate_2_3<4>();
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG2, 0);
    // LREG15 is the documented read-only vector [0,2,...,62]. Its low
    // four bits are zero precisely in lane0 of each eight-lane face row.
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_USHORT, 14);
    TTI_SFPAND(0, 15, p_sfpu::LREG4, 0);
    TTI_SFPPUSHC(0, 0, 0, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSETCC(0, p_sfpu::LREG4, 0, 2);  // non-lane0
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    TTI_SFPPOPC(0, 0, 0, 0);
    adaptive_max_broadcast();  // scores nonnegative, other lanes are zero
}

template <int BASE, int OFFSET, bool FIRST>
inline void adaptive_candidate() {
    TTI_SFPLOAD(p_sfpu::LREG0, 3, ADDR_MOD_7, BASE);
    TTI_SFPLOAD(p_sfpu::LREG1, 3, ADDR_MOD_7, BASE + 2);
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 1);
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 1);
    TTI_SFPLOAD(p_sfpu::LREG2, 3, ADDR_MOD_7, BASE + 130);  // 2^E
    TTI_SFPEXEXP(0, p_sfpu::LREG2, p_sfpu::LREG4, 1);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_USHORT, 254);
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG4, 6);  // 254 - biased E
    TTI_SFPSETEXP(0, p_sfpu::LCONST_1, p_sfpu::LREG4, 0);  // 2^-E
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    // After exact power-of-two normalization, maximum is in [1,2).
    // Fixed constants implement RNE grids 1/4, 1/8, or 1/2 and their caps.
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_FLOATB, (148 + OFFSET) << 7);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0x3fe0 + OFFSET * 128);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG6, 0);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::LREG5, 2);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LREG6, 2);
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG7, 0);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG5, 1);
    TTI_SFPNOP;
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, 1);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG0, 2);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG1, 2);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    adaptive_mse_reduce();
    if constexpr (!FIRST) {
        TTI_SFPLOAD(p_sfpu::LREG3, 3, ADDR_MOD_7, BASE + 128);
        TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG4, 2);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPSETCC(0, p_sfpu::LREG4, 0, 0);  // candidate < best; ties keep baseline
    }
    TTI_SFPSTORE(p_sfpu::LREG5, 3, ADDR_MOD_7, BASE + 64);
    TTI_SFPSTORE(p_sfpu::LREG6, 3, ADDR_MOD_7, BASE + 66);
    TTI_SFPSTORE(p_sfpu::LREG2, 3, ADDR_MOD_7, BASE + 128);
    if constexpr (!FIRST) {
        TTI_SFPPOPC(0, 0, 0, 0);
    }
}

template <int SEARCH, int BASE>
inline void adaptive_four_rows() {
    TTI_SFPLOAD(p_sfpu::LREG0, 3, ADDR_MOD_7, BASE);
    TTI_SFPLOAD(p_sfpu::LREG1, 3, ADDR_MOD_7, BASE + 2);
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG2, 1);
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG3, 1);
    adaptive_max_2_3();
    adaptive_max_broadcast();
    TTI_SFPSETEXP(0, p_sfpu::LCONST_1, p_sfpu::LREG2, 2);
    TTI_SFPSTORE(p_sfpu::LREG2, 3, ADDR_MOD_7, BASE + 130);
    adaptive_candidate<BASE, 0, true>();
    if constexpr (SEARCH >= 1) {
        adaptive_candidate<BASE, -1, false>();
    }
    if constexpr (SEARCH >= 2) {
        adaptive_candidate<BASE, 1, false>();
    }
    TTI_SFPLOAD(p_sfpu::LREG0, 3, ADDR_MOD_7, BASE);
    TTI_SFPLOAD(p_sfpu::LREG1, 3, ADDR_MOD_7, BASE + 2);
    TTI_SFPLOAD(p_sfpu::LREG2, 3, ADDR_MOD_7, BASE + 130);
    TTI_SFPLOAD(p_sfpu::LREG5, 3, ADDR_MOD_7, BASE + 64);
    TTI_SFPLOAD(p_sfpu::LREG6, 3, ADDR_MOD_7, BASE + 66);
    TTI_SFPMUL(p_sfpu::LREG5, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG5, 0);
    TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
    TTI_SFPSETSGN(0, p_sfpu::LREG5, p_sfpu::LREG0, 0);
    TTI_SFPSETSGN(0, p_sfpu::LREG6, p_sfpu::LREG1, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, 3, ADDR_MOD_7, BASE);
    TTI_SFPSTORE(p_sfpu::LREG1, 3, ADDR_MOD_7, BASE + 2);
}

template <int SEARCH>
inline void adaptive_face() {
    adaptive_four_rows<SEARCH, 0>();
    adaptive_four_rows<SEARCH, 4>();
    adaptive_four_rows<SEARCH, 8>();
    adaptive_four_rows<SEARCH, 12>();
}
}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    constexpr uint32_t search = get_compile_time_arg_val(0);
    static_assert(DST_ACCUM_MODE, "Adaptive MSE requires FP32 DST scratch");
    static_assert(search <= 2);
    const uint32_t count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 16);
    copy_init(0);
    negative_tile_init();
    MATH((ckernel::math::_configure_src_zero_flag_(false)));
    for (uint32_t i = 0; i < count; ++i) {
        cb_wait_front(0, 1);
        cb_reserve_back(16, 1);
        tile_regs_acquire();
        copy_tile(0, 0, 0);
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, adaptive_face, (search), 0, VectorMode::RC));
        tile_regs_commit();
        cb_pop_front(0, 1);
        tile_regs_wait();
        pack_tile(0, 16);
        tile_regs_release();
        cb_push_back(16, 1);
    }
}
