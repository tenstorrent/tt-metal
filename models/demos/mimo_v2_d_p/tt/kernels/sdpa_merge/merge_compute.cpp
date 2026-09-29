// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// K-split SDPA merge, compute (fp32 DEST, full sync: 8 tiles). Per work item, partitions p < S with row max m_p
// (column 0 of the max tile), running sum l_p (row sum of the 32 per-column partials) and unnormalized output O_p:
//   M = max_p m_p, a_p = exp(scale (m_p - M)), coef_p = a_p / sum_q a_q l_q, out = sum_p coef_p O_p.
// Two DEST sessions per item, no fp32 round trips through L1:
//   1. m_p (copy) and L_p = l_p x ones (matmul: every column the row sum) into DEST, the coefficients by SFPU in
//      place (m: D[0, S), L then a L: D[S, 2S), M then 1 / den: D[2S]) -> packed to c_25 (fp32)
//   2. per group of G output column tiles: coef_p (column broadcast) * O_p for every partition into DEST, summed by
//      SFPU adds, packed as bf16 output
// Items alternate between two data movement lanes (merge_dm.cpp): item k's CBs are lane k % 2's
// (max c_(4l), sum c_(1+4l), O c_(2+4l), out c_(16+l)); c_3 all-ones tile, c_25 coefficients.
// CT: 0 S, 1 DVt, 2 scale (fp32 bits)   RT: 0 items on this core
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#ifdef MERGE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define MZ(name) DeviceZoneScopedN(name)
#else
#define MZ(name)
#endif

constexpr uint32_t CB_ONE = tt::CBIndex::c_3, CB_COEF = tt::CBIndex::c_25;

// The coefficient math is per row: only column 0 of each stats tile matters, so the SFPU works on the left column
// of faces only (VectorMode::C: columns 0-15, half the work of a full tile). Same calls as the compute API wrappers.
ALWI void sub_c(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::SUB, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        o,
        VectorMode::C)));
}
ALWI void add_c(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        o,
        VectorMode::C)));
}
ALWI void mul_c(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE),
        a,
        b,
        o,
        VectorMode::C)));
}
ALWI void mul_scalar_c(uint32_t i, uint32_t bits) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, MUL_UNARY, 8, DST_ACCUM_MODE),
        i,
        VectorMode::C,
        bits));
}
constexpr uint32_t DST = 8;  // fp32 DEST tiles, full sync

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0), DVt = get_compile_time_arg_val(1);
    constexpr uint32_t scale_bits = get_compile_time_arg_val(2);
    static_assert(2 * S + 1 <= DST && S <= DST, "session 1 needs 2 S + 1 DEST tiles");
    constexpr uint32_t G = DST / S < DVt ? DST / S : DVt;  // output column tiles per session-2 group
    const uint32_t mine = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(tt::CBIndex::c_0, CB_ONE, CB_COEF);
    cb_wait_front(CB_ONE, 1);
    for (uint32_t k = 0; k < mine; ++k) {
        const uint32_t lane = k & 1;
        const uint32_t CB_M = 4 * lane, CB_L = 1 + 4 * lane, CB_O = 2 + 4 * lane, CB_OUT = 16 + lane;
#ifdef MERGE_DM_ONLY
        // debug: data movement alone (no math)
        cb_wait_front(CB_M, S);
        cb_wait_front(CB_L, S);
        cb_wait_front(CB_O, S * DVt);
        cb_pop_front(CB_M, S);
        cb_pop_front(CB_L, S);
        cb_pop_front(CB_O, S * DVt);
        cb_reserve_back(CB_OUT, DVt);
        cb_push_back(CB_OUT, DVt);
        continue;
#endif

        // 1. coefficients
        {
            MZ("MG_C_WAITIN");
            cb_wait_front(CB_M, S);
            cb_wait_front(CB_L, S);
            cb_wait_front(CB_O, S * DVt);
        }
        {
            MZ("MG_S1");
            cb_reserve_back(CB_COEF, S);
            tile_regs_acquire();
            {
                MZ("MG_S1_LOAD");
                reconfig_data_format(CB_M, CB_ONE);
                copy_init(CB_M);
                for (uint32_t p = 0; p < S; ++p) {
                    copy_tile(CB_M, p, p);
                }
                matmul_init(CB_L, CB_ONE);
                for (uint32_t p = 0; p < S; ++p) {
                    matmul_tiles(CB_L, CB_ONE, p, 0, S + p);  // every column = the row sum of the 32 partials
                }
            }
            {
                MZ("MG_S1_MAXSUB");
                binary_max_tile_init();
                binary_max_tile(0, S > 1 ? 1 : 0, 2 * S, VectorMode::C);
                for (uint32_t p = 2; p < S; ++p) {
                    binary_max_tile(2 * S, p, 2 * S, VectorMode::C);
                }
                sub_binary_tile_init();
                for (uint32_t p = 0; p < S; ++p) {
                    sub_c(p, 2 * S, p);
                }
                binop_with_scalar_tile_init();
                for (uint32_t p = 0; p < S; ++p) {
                    mul_scalar_c(p, scale_bits);
                }
            }
            {
                MZ("MG_S1_EXP");
                exp_tile_init<false>();
                for (uint32_t p = 0; p < S; ++p) {
                    exp_tile<false>(p, VectorMode::C);  // a_p
                }
            }
            {
                MZ("MG_S1_DEN");
                mul_binary_tile_init();
                for (uint32_t p = 0; p < S; ++p) {
                    mul_c(p, S + p, S + p);  // a_p L_p
                }
                add_binary_tile_init();
                for (uint32_t p = 1; p < S; ++p) {
                    add_c(S, S + p, S);  // den
                }
            }
            {
                MZ("MG_S1_RECIP");
                recip_tile_init();
                recip_tile(S, VectorMode::C);
            }
            mul_binary_tile_init();
            for (uint32_t p = 0; p < S; ++p) {
                mul_c(p, S, p);  // coef_p
            }
            {
                MZ("MG_S1_SFPU_END");
                tile_regs_commit();
            }
            tile_regs_wait();
            pack_reconfig_data_format(CB_COEF);
            for (uint32_t p = 0; p < S; ++p) {
                pack_tile(p, CB_COEF);
            }
            tile_regs_release();
            cb_push_back(CB_COEF, S);
            cb_pop_front(CB_M, S);
            cb_pop_front(CB_L, S);
        }
        MZ("MG_S2");

        // 2. out = sum_p coef_p * O_p, G column tiles at a time: partition p's products in D[p G, p G + G)
        cb_wait_front(CB_COEF, S);
        cb_wait_front(CB_O, S * DVt);
        cb_reserve_back(CB_OUT, DVt);
        for (uint32_t j0 = 0; j0 < DVt; j0 += G) {
            tile_regs_acquire();
            reconfig_data_format(CB_O, CB_COEF);
            mul_bcast_cols_init(CB_O, CB_COEF);
            for (uint32_t p = 0; p < S; ++p) {
                for (uint32_t j = 0; j < G; ++j) {
                    mul_tiles_bcast_cols(CB_O, CB_COEF, p * DVt + j0 + j, p, p * G + j);
                }
            }
            if constexpr (S > 1) {
                add_binary_tile_init();
                for (uint32_t p = 1; p < S; ++p) {
                    for (uint32_t j = 0; j < G; ++j) {
                        add_binary_tile(j, p * G + j, j);
                    }
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_OUT);
            for (uint32_t j = 0; j < G; ++j) {
                pack_tile(j, CB_OUT);
            }
            tile_regs_release();
        }
        cb_push_back(CB_OUT, DVt);
        cb_pop_front(CB_COEF, S);
        cb_pop_front(CB_O, S * DVt);
    }
}
