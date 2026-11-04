// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Ternary decode linear v3, compute. y[n] = sum_kb s[n, kb] * (x'_kb @ T_kb)[n], kb = 128-wide K blocks.
// Per output tile column and per kb: custom_mm (in0 [1,32] tiles of x', in1 the 4 ternary tiles of kb as Bfp2_b with
// a forced exponent, LoFi) leaves P in DEST tile 1, rows 0 and 16; the kb's fp16 scales, a [1,32] tile, are copied
// into DEST tile 2 at the same rows; the SFPU adds P * s into tile 0 and zeroes P (both in fp32).
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/pack.h"
#include "api/debug/kernel_profiler.hpp"

constexpr uint32_t cb_x = tt::CBIndex::c_0;
constexpr uint32_t cb_w0 = tt::CBIndex::c_1;  // even records (NCRISC)
constexpr uint32_t cb_s = tt::CBIndex::c_2;
constexpr uint32_t cb_w1 = tt::CBIndex::c_6;  // odd records (BRISC)
constexpr uint32_t cb_out = tt::CBIndex::c_16;
constexpr auto LAYOUT = CustomMmDestLayout::Default;
#ifndef SFPU_ITER
#define SFPU_ITER 32
#endif

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// acc += p * s; p = 0, over the first ITER SFPU rows (two DEST rows each) of each tile: rows 0 and 16 hold a
// [1,32] tile's two face rows.
template <int ITER, bool FIRST>
inline void calculate_scale_acc(uint32_t acc, uint32_t p, uint32_t s, uint32_t) {
    constexpr uint32_t T = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITER; ++d) {
        sfpi::vFloat pv = sfpi::dst_reg[p * T];
        sfpi::vFloat sv = sfpi::dst_reg[s * T];
        if constexpr (FIRST) {
            sfpi::dst_reg[acc * T] = pv * sv;
        } else {
            sfpi::vFloat av = sfpi::dst_reg[acc * T];
            sfpi::dst_reg[acc * T] = pv * sv + av;
        }
        sfpi::dst_reg[p * T] = 0.0f;
        sfpi::dst_reg++;
    }
}
// SFPU geometry of an fp32 DEST tile (measured, tests/probe4.py): face f (16x16) is SFPU rows 8f..8f+7; SFPU row 8f+k
// holds face rows 4*(k/2)..4*(k/2)+3 (one per 8-lane subvector) at the even (k even) or odd (k odd) columns.
// P, a [1,32] result, is face row 0 of faces 0 and 1: subvector 0 of SFPU rows 0, 1, 8, 9. The scale tile holds 32
// kbs, kb j at tile row j: subvector j % 4 of SFPU rows SB, SB+1, SB+8, SB+9, SB = (j < 16 ? 0 : 16) + 2 * ((j % 16) /
// 4). subvec_transp brings subvector j % 4 of the loaded scale row to subvector 0 of register j % 4.
template <int J, bool FIRST>
inline void scale_acc_kb32(uint32_t acc, uint32_t p, uint32_t s, uint32_t) {
    constexpr uint32_t T = 32;
    constexpr uint32_t SB = (J < 16 ? 0 : 16) + 2 * ((J % 16) / 4);
    constexpr uint32_t OFF[4] = {0, 1, 8, 9};
#pragma GCC unroll 4
    for (int h = 0; h < 4; ++h) {
        sfpi::vFloat a = sfpi::dst_reg[s * T + SB + OFF[h]];
        sfpi::vFloat b = 0.0f, c = 0.0f, d = 0.0f;
        sfpi::subvec_transp(a, b, c, d);
        sfpi::vFloat sv = (J % 4 == 0) ? a : (J % 4 == 1) ? b : (J % 4 == 2) ? c : d;
        sfpi::vFloat pv = sfpi::dst_reg[p * T + OFF[h]];
        if constexpr (FIRST) {
            sfpi::dst_reg[acc * T + OFF[h]] = pv * sv;
        } else {
            sfpi::vFloat av = sfpi::dst_reg[acc * T + OFF[h]];
            sfpi::dst_reg[acc * T + OFF[h]] = pv * sv + av;
        }
        sfpi::dst_reg[p * T + OFF[h]] = 0.0f;
    }
}
// Comma-free names for the SFPU call macro.
inline void scale_acc_first(uint32_t a, uint32_t p, uint32_t s, uint32_t o) {
    calculate_scale_acc<SFPU_ITER, true>(a, p, s, o);
}
inline void scale_acc_next(uint32_t a, uint32_t p, uint32_t s, uint32_t o) {
    calculate_scale_acc<SFPU_ITER, false>(a, p, s, o);
}
// clang-format off
#define SCALE_ROW_FNS(J)                                                                                              \
    inline void sr_first_##J(uint32_t a, uint32_t p, uint32_t s, uint32_t o) { scale_acc_kb32<J, true>(a, p, s, o); } \
    inline void sr_next_##J(uint32_t a, uint32_t p, uint32_t s, uint32_t o) { scale_acc_kb32<J, false>(a, p, s, o); }
SCALE_ROW_FNS(0)
SCALE_ROW_FNS(1)
SCALE_ROW_FNS(2)
SCALE_ROW_FNS(3)
SCALE_ROW_FNS(4) SCALE_ROW_FNS(5) SCALE_ROW_FNS(6) SCALE_ROW_FNS(7) SCALE_ROW_FNS(8) SCALE_ROW_FNS(9) SCALE_ROW_FNS(10)
    SCALE_ROW_FNS(11) SCALE_ROW_FNS(12) SCALE_ROW_FNS(13) SCALE_ROW_FNS(14) SCALE_ROW_FNS(15) SCALE_ROW_FNS(16)
        SCALE_ROW_FNS(17) SCALE_ROW_FNS(18) SCALE_ROW_FNS(19) SCALE_ROW_FNS(20) SCALE_ROW_FNS(21) SCALE_ROW_FNS(22)
            SCALE_ROW_FNS(23) SCALE_ROW_FNS(24) SCALE_ROW_FNS(25) SCALE_ROW_FNS(26) SCALE_ROW_FNS(27) SCALE_ROW_FNS(28)
                SCALE_ROW_FNS(29) SCALE_ROW_FNS(30) SCALE_ROW_FNS(31)
// clang-format on
}  // namespace ckernel::sfpu
#endif

ALWI void ternary_srca() {
    UNPACK(({
        _llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(
            (uint32_t)DataFormat::Bfp2_b, (uint32_t)DataFormat::Float16_b, 256 >> 4, FACE_R_DIM, 4);
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(1);
        cfg_reg_rmw_tensix<UNP0_FORCED_SHARED_EXP_shared_exp_RMW>(127);
    }));
}

#ifndef SFPU_REINIT
#define SFPU_REINIT 1
#endif
#ifndef SCALE_FP32
#define SCALE_FP32 0
#endif
ALWI void scale_tile_srca() {
    // A 32x32 IEEE fp16 tile (2048 B) in a CB declared Float32 (srcA Tf32, which holds fp16 exactly).
    UNPACK(({
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(0);
        _llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::FACE_ROW_MAJOR>(
            (uint32_t)DataFormat::Float16, (uint32_t)DataFormat::Tf32, 2048 >> 4, FACE_R_DIM, 4);
    }));
}

// clang-format off
#ifdef TRISC_MATH
#define SR_CASE(J)                                                                          \
    case J:                                                                                 \
        if (first) {                                                                        \
            SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(                                             \
                DST_SYNC_MODE, DST_ACCUM_MODE, sr_first_##J, 0, 1, 2, 0, VectorMode::None); \
        } else {                                                                            \
            SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(                                             \
                DST_SYNC_MODE, DST_ACCUM_MODE, sr_next_##J, 0, 1, 2, 0, VectorMode::None);  \
        }                                                                                   \
        break;
#endif
ALWI void scale_acc_kb(uint32_t j, bool first) {
#ifdef TRISC_MATH
    switch (j) {
        SR_CASE(0)
        SR_CASE(1)
        SR_CASE(2)
        SR_CASE(3)
        SR_CASE(4) SR_CASE(5) SR_CASE(6) SR_CASE(7) SR_CASE(8) SR_CASE(9) SR_CASE(10) SR_CASE(11) SR_CASE(12)
            SR_CASE(13) SR_CASE(14) SR_CASE(15) SR_CASE(16) SR_CASE(17) SR_CASE(18) SR_CASE(19) SR_CASE(20) SR_CASE(21)
                SR_CASE(22) SR_CASE(23) SR_CASE(24) SR_CASE(25) SR_CASE(26) SR_CASE(27) SR_CASE(28) SR_CASE(29)
                    SR_CASE(30) SR_CASE(31) default : break;
    }
#endif
}
// clang-format on

ALWI void scale_srca() {
    if constexpr (SCALE_FP32) {
        UNPACK((cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(0)));
        return;
    }
    // cb_s is declared Float32 (srcA Tf32); its [1,32] tiles are IEEE fp16 (64 B), which Tf32 holds exactly.
    UNPACK(({
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Force_shared_exp_RMW>(0);
        // FACE_ROW_MAJOR: also reprogram the face geometry, so face 1 is read 32 B after face 0 (fp16), not 64 B
        // (the declared fp32 format's stride).
        _llk_unpack_reconfig_data_format_srca_impl_<DST_ACCUM_MODE, p_dim_stride_target::FACE_ROW_MAJOR>(
            (uint32_t)DataFormat::Float16, (uint32_t)DataFormat::Tf32, 64 >> 4, 1, 2);
    }));
}

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t KB = get_compile_time_arg_val(1);
    const uint32_t num_nt = get_arg_val<uint32_t>(0);

    {
        DeviceZoneScopedN("TD-START");
        custom_mm_block_init<false, LAYOUT>(cb_x, cb_w0, cb_out);
#if !SFPU_REINIT
        addcmul_tile_init();
#endif
        cb_wait_front(cb_x, Kt);
#if defined(INLINE_SCALES)
#elif defined(SCALE_GROUP16)
        cb_wait_front(cb_s, num_nt * ((KB + 31) / 32));
#else
        cb_wait_front(cb_s, num_nt * KB);
#endif
    }

#ifdef SCALE_GROUP16
    // Per output tile column: per group of 32 kbs one fp16 scale tile into DEST tile 2 (kb j of the group at tile row
    // j) and one custom_mm setup; per kb custom_mm into DEST tile 1, then acc (tile 0) += P * S on two SFPU
    // rows. Scale tiles: G = ceil(KB / 32) per output tile column.
    constexpr uint32_t G = (KB + 31) / 32;
    for (uint32_t i = 0; i < num_nt; ++i) {
        DeviceZoneScopedN("TD-NT");
        const uint32_t cb_w = (i & 1) ? cb_w1 : cb_w0;
        tile_regs_acquire();
        for (uint32_t g = 0; g < G; ++g) {
            {
                DeviceZoneScopedN("TD-GSETUP");
#ifdef INLINE_SCALES
                // The group's scale tile: the next 8 pages of the record (2048 B).
                cb_wait_front(cb_w, 8);
                copy_init(cb_s);
                scale_tile_srca();
                copy_tile(cb_w, 0, 2);
                cb_pop_front(cb_w, 8);
#else
                copy_init(cb_s);
                scale_tile_srca();
                copy_tile(cb_s, i * G + g, 2);
#endif
                reconfig_data_format_srca(cb_w);
                custom_mm_block_init_short<false, LAYOUT>(cb_x, cb_w, cb_out);
                ternary_srca();
                addcmul_tile_init();
            }
            const uint32_t nj = (KB - g * 32 < 32) ? (KB - g * 32) : 32;
            for (uint32_t j = 0; j < nj; ++j) {
                const uint32_t kb = g * 32 + j;
                if ((kb & 3) == 0) {
                    cb_wait_front(cb_w, 16);
                }
#ifndef SKIP_MM
                custom_mm_block<LAYOUT, false>(cb_x, cb_w, kb * 4, (kb & 3) * 4, 1, 4);
#endif
#ifdef DBG_FIRST_KB
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_out, 2);
                pack_reconfig_data_format(cb_out);
                pack_tile(1, cb_out);
                pack_tile(2, cb_out);
                cb_push_back(cb_out, 2);
                tile_regs_release();
                tile_regs_acquire();
                copy_init(cb_s);
                scale_tile_srca();
                copy_tile(cb_s, i * G + g, 2);
                reconfig_data_format_srca(cb_w);
                custom_mm_block_init_short<false, LAYOUT>(cb_x, cb_w, cb_out);
                ternary_srca();
                addcmul_tile_init();
                custom_mm_block<LAYOUT, false>(cb_x, cb_w, kb * 4, (kb & 3) * 4, 1, 4);
                scale_acc_kb(j, kb == 0);
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_out, 1);
                pack_tile(0, cb_out);
                cb_push_back(cb_out, 1);
                tile_regs_release();
                return;
#endif
#ifndef SKIP_SFPU
                scale_acc_kb(j, kb == 0);
                MATH((llk_math_custom_mm_init<false, LAYOUT>(cb_x, cb_w, 1)));
#endif
                if ((kb & 3) == 3) {
                    cb_pop_front(cb_w, 16);
                }
            }
        }
#ifdef INLINE_SCALES
        constexpr uint32_t pad = RECORD_PAGES - Kt - G * 8;
        if constexpr (pad > 0) {
            cb_wait_front(cb_w, pad);
            cb_pop_front(cb_w, pad);
        }
#endif
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
#else
    // Per output tile column, per 4-KB page of 4 kbs: one scale copy setup and the page's 4 scale tiles into DEST tiles
    // 4..7, one custom_mm setup, then per kb custom_mm into DEST tile 1 and the SFPU's acc (tile 0) += P * S.
    // DEST is full-sync (8 fp32 tiles).
    for (uint32_t i = 0; i < num_nt; ++i) {
        const uint32_t cb_w = (i & 1) ? cb_w1 : cb_w0;
        tile_regs_acquire();
        for (uint32_t g = 0; g < KB / 4; ++g) {
            cb_wait_front(cb_w, 16);
            copy_init(cb_s);
            scale_srca();
            for (uint32_t j = 0; j < 4; ++j) {
                copy_tile(cb_s, i * KB + g * 4 + j, 4 + j);
            }
            reconfig_data_format_srca(cb_w);
            custom_mm_block_init_short<false, LAYOUT>(cb_x, cb_w, cb_out);
            ternary_srca();
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t kb = g * 4 + j;
#ifndef SKIP_MM
                custom_mm_block<LAYOUT, false>(cb_x, cb_w, kb * 4, j * 4, 1, 4);
#endif
#ifndef SKIP_SFPU
#if SFPU_REINIT
                addcmul_tile_init();
#endif
                if (kb == 0) {
                    MATH((SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(
                        DST_SYNC_MODE, DST_ACCUM_MODE, scale_acc_first, 0, 1, 4 + j, 0, VectorMode::None)));
                } else {
                    MATH((SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(
                        DST_SYNC_MODE, DST_ACCUM_MODE, scale_acc_next, 0, 1, 4 + j, 0, VectorMode::None)));
                }
#endif
#if !SFPU_REINIT
                // custom_mm's math init owns the address modifiers; restore its state for the next kb.
                MATH((llk_math_custom_mm_init<false, LAYOUT>(cb_x, cb_w, 1)));
#endif
            }
            cb_pop_front(cb_w, 16);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
#endif
    custom_mm_block_uninit<LAYOUT>(cb_out);
}
