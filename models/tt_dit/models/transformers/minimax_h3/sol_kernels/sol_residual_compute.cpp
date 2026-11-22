// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Block residual compute (sol_prepass.block_residual_reference), one unit = (head, local Q tile row), fp32 in DST:
//   lg  = scale * (q_bar . kc + ln len / scale + mask * (-RES_MASK / scale))   three matmuls per key tile
//   m   = row max of lg (bf16 copy; any shift close to the max works)
//   p   = exp(lg - m), den = sum p (fp32: elementwise in DST, the row sum of its bf16 hi / lo parts by matmul)
//   res = (bf16(p) @ vc) / den, written twice per Q block (the two pooled entries) by a selection matmul
//   w   = (m + ln den) / scale clamped at NEG, split into two bf16-exact column biases (residual_bias_split).
// A Q block with no residual has den = 0: ln 0 = -inf gives the NEG bias, the clamped den a zero residual.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/reg_api.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// Truncate fp32 to bf16 (toward zero): clear the low 16 bits.
template <int ITERATIONS = 8>
inline void calculate_trunc_bf16() {
    for (int d = 0; d < ITERATIONS; ++d) {
        sfpi::vInt bits = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[0]));
        sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(bits & sfpi::vInt(-65536));
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

inline void trunc_bf16_tile(uint32_t idst) {
    SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_trunc_bf16, (8 /* ITERATIONS */), idst, VectorMode::RC);
}
#endif

namespace {
constexpr uint32_t cb_q = 0, cb_kc = 1, cb_ka = 2, cb_mk = 3, cb_vc = 4, cb_tiles = 5, cb_lg = 6, cb_lg16 = 7, cb_m = 8,
                   cb_acc = 9, cb_ahi = 10, cb_alo = 11, cb_den = 12, cb_rden = 13, cb_p16 = 14, cb_res = 15,
                   cb_pv = 16, cb_bias = 17;
// cb_tiles: q_aug (columns 0, 1 = 1), diag(-RES_MASK / scale), ones, the two row-duplication selections.
constexpr uint32_t T_QAUG = 0, T_DNEG = 1, T_ONES = 2, T_DUP = 3;
constexpr uint32_t B = 4;               // key tiles per DST batch of the softmax
constexpr uint32_t ONE = 0x3F800000u;   // 1.0f
constexpr uint32_t TINY = 0x0DA24260u;  // 1e-30f

// Matmul and reduce accumulate into DST, whose contents are not cleared for us: set the slots first.
inline void fill_dst(uint32_t first, uint32_t count, float value) {
    fill_tile_init();
    for (uint32_t i = first; i < first + count; ++i) {
        fill_tile(i, value);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t NBGT = get_compile_time_arg_val(2);
    constexpr uint32_t DT = get_compile_time_arg_val(3);
    constexpr uint32_t NX = get_compile_time_arg_val(4);
    constexpr uint32_t NY = get_compile_time_arg_val(5);
    constexpr uint32_t UNITS = get_compile_time_arg_val(6);
    constexpr uint32_t KCH = get_compile_time_arg_val(7);
    constexpr uint32_t SCALE = get_compile_time_arg_val(8);
    constexpr uint32_t INV_SCALE = get_compile_time_arg_val(9);
    constexpr uint32_t NEG = get_compile_time_arg_val(10);
    static_assert(2 * DT <= 8 && KCH <= 8 && B + 2 <= 8, "fp32 full-sync DST holds 8 tiles");
    static_assert(NBGT % KCH == 0 && NBGT % B == 0, "key tiles come in whole chunks");

    constexpr uint32_t NCORES = NX * NY;
    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    const uint32_t unit_start = ci * UNITS / NCORES;
    const uint32_t unit_count = (ci + 1) * UNITS / NCORES - unit_start;
    compute_kernel_hw_startup(cb_q, cb_kc, cb_lg);
    if (unit_count == 0) {
        return;
    }
    CircularBuffer q(cb_q), kc(cb_kc), ka(cb_ka), mk(cb_mk), vc(cb_vc), tiles(cb_tiles), lg(cb_lg), lg16(cb_lg16),
        m(cb_m), acc(cb_acc), ahi(cb_ahi), alo(cb_alo), den(cb_den), rden(cb_rden), p16(cb_p16), res(cb_res), pv(cb_pv),
        bias(cb_bias);
    tiles.wait_front(5);

    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        q.wait_front(DT);

        // Logits, fp32 (cb_lg) and a bf16 copy for the row max (cb_lg16).
        lg.reserve_back(NBGT);
        lg16.reserve_back(NBGT);
        for (uint32_t c0 = 0; c0 < NBGT; c0 += KCH) {
            kc.wait_front(KCH * DT);
            ka.wait_front(KCH);
            mk.wait_front(KCH);
            tile_regs_acquire();
            fill_dst(0, KCH, 0.0f);
            reconfig_data_format(cb_q, cb_kc);
            matmul_init(cb_q, cb_kc, 1);
            for (uint32_t cc = 0; cc < KCH; ++cc) {
                for (uint32_t n = 0; n < DT; ++n) {
                    matmul_tiles(cb_q, cb_kc, n, cc * DT + n, cc);
                }
            }
            reconfig_data_format(cb_tiles, cb_ka);
            matmul_init(cb_tiles, cb_ka, 1);
            for (uint32_t cc = 0; cc < KCH; ++cc) {
                matmul_tiles(cb_tiles, cb_ka, T_QAUG, cc, cc);
            }
            reconfig_data_format(cb_mk, cb_tiles);
            matmul_init(cb_mk, cb_tiles, 1);
            for (uint32_t cc = 0; cc < KCH; ++cc) {
                matmul_tiles(cb_mk, cb_tiles, cc, T_DNEG, cc);
            }
            binop_with_scalar_tile_init();
            for (uint32_t cc = 0; cc < KCH; ++cc) {
                mul_unary_tile(cc, SCALE);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_lg);
            for (uint32_t cc = 0; cc < KCH; ++cc) {
                pack_tile<true>(cc, cb_lg, c0 + cc);
            }
            pack_reconfig_data_format(cb_lg16);
            for (uint32_t cc = 0; cc < KCH; ++cc) {
                pack_tile<true>(cc, cb_lg16, c0 + cc);
            }
            tile_regs_release();
            kc.pop_front(KCH * DT);
            ka.pop_front(KCH);
            mk.pop_front(KCH);
        }
        lg.push_back(NBGT);
        lg16.push_back(NBGT);

        // Row max (column 0 of cb_m).
        lg16.wait_front(NBGT);
        m.reserve_back(1);
        tile_regs_acquire();
        fill_dst(0, 1, -3.0e38f);
        reconfig_data_format(cb_lg16, cb_tiles);
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_lg16, cb_tiles, cb_m);
        for (uint32_t c = 0; c < NBGT; ++c) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_lg16, cb_tiles, c, T_ONES, 0);
        }
        reduce_uninit();
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_m);
        pack_tile(0, cb_m);
        tile_regs_release();
        m.push_back(1);
        lg16.pop_front(NBGT);

        // p = exp(lg - m) (bf16 to cb_p16), the fp32 elementwise sum carried in cb_acc.
        lg.wait_front(NBGT);
        m.wait_front(1);
        p16.reserve_back(NBGT);
        for (uint32_t c0 = 0; c0 < NBGT; c0 += B) {
            tile_regs_acquire();
            reconfig_data_format_srca(cb_lg);
            copy_init(cb_lg);
            for (uint32_t i = 0; i < B; ++i) {
                copy_tile(cb_lg, c0 + i, i);
            }
            copy_init(cb_m);
            copy_tile(cb_m, 0, B);
            if (c0 > 0) {
                acc.wait_front(1);
                copy_init(cb_acc);
                copy_tile(cb_acc, 0, B + 1);
            } else {
                fill_dst(B + 1, 1, 0.0f);
            }
            sfpu_bcast_col_init();
            for (uint32_t i = 0; i < B; ++i) {
                sfpu_sub_bcast_col(i, B);
            }
            exp_tile_init();
            for (uint32_t i = 0; i < B; ++i) {
                exp_tile(i);
            }
            add_binary_tile_init();
            for (uint32_t i = 0; i < B; ++i) {
                add_binary_tile(B + 1, i, B + 1);
            }
            tile_regs_commit();
            if (c0 > 0) {
                acc.pop_front(1);
            }
            acc.reserve_back(1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_p16);
            for (uint32_t i = 0; i < B; ++i) {
                pack_tile<true>(i, cb_p16, c0 + i);
            }
            pack_reconfig_data_format(cb_acc);
            pack_tile(B + 1, cb_acc);
            tile_regs_release();
            acc.push_back(1);
        }
        p16.push_back(NBGT);
        lg.pop_front(NBGT);

        // den = row sum of acc: its bf16 hi and lo parts times ones; rden = 1 / max(den, TINY).
        acc.wait_front(1);
        ahi.reserve_back(1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_acc);
        copy_init(cb_acc);
        copy_tile(cb_acc, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_ahi);
        pack_tile(0, cb_ahi);
        tile_regs_release();
        ahi.push_back(1);
        ahi.wait_front(1);
        alo.reserve_back(1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_acc);
        copy_init(cb_acc);
        copy_tile(cb_acc, 0, 0);
        reconfig_data_format_srca(cb_ahi);
        copy_init(cb_ahi);
        copy_tile(cb_ahi, 0, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_alo);
        pack_tile(0, cb_alo);
        tile_regs_release();
        alo.push_back(1);
        acc.pop_front(1);
        alo.wait_front(1);
        den.reserve_back(1);
        rden.reserve_back(1);
        tile_regs_acquire();
        fill_dst(0, 1, 0.0f);
        reconfig_data_format(cb_ahi, cb_tiles);
        matmul_init(cb_ahi, cb_tiles, 0);
        matmul_tiles(cb_ahi, cb_tiles, 0, T_ONES, 0);
        matmul_tiles(cb_alo, cb_tiles, 0, T_ONES, 0);
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(0, 1);
        unary_max_tile_init();
        unary_max_tile(1, TINY);
        recip_tile_init();
        recip_tile(1);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_den);
        pack_tile(0, cb_den);
        pack_tile(1, cb_rden);
        tile_regs_release();
        den.push_back(1);
        rden.push_back(1);
        ahi.pop_front(1);
        alo.pop_front(1);

        // res = (bf16(p) @ vc) * rden.
        p16.wait_front(NBGT);
        rden.wait_front(1);
        res.reserve_back(DT);
        tile_regs_acquire();
        fill_dst(0, DT, 0.0f);
        reconfig_data_format(cb_p16, cb_vc);
        matmul_init(cb_p16, cb_vc, 0);
        for (uint32_t c0 = 0; c0 < NBGT; c0 += KCH) {
            vc.wait_front(KCH * DT);
            for (uint32_t cc = 0; cc < KCH; ++cc) {
                for (uint32_t n = 0; n < DT; ++n) {
                    matmul_tiles(cb_p16, cb_vc, c0 + cc, cc * DT + n, n);
                }
            }
            vc.pop_front(KCH * DT);
        }
        reconfig_data_format_srca(cb_rden);
        copy_init(cb_rden);
        copy_tile(cb_rden, 0, DT);
        sfpu_bcast_col_init();
        for (uint32_t n = 0; n < DT; ++n) {
            sfpu_mul_bcast_col(n, DT);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_res);
        for (uint32_t n = 0; n < DT; ++n) {
            pack_tile(n, cb_res);
        }
        tile_regs_release();
        res.push_back(DT);
        p16.pop_front(NBGT);
        rden.pop_front(1);

        // The two pooled entries per Q block: output tile row t holds Q blocks 16t..16t+15, each twice.
        res.wait_front(DT);
        pv.reserve_back(2 * DT);
        tile_regs_acquire();
        fill_dst(0, 2 * DT, 0.0f);
        reconfig_data_format(cb_tiles, cb_res);
        matmul_init(cb_tiles, cb_res, 0);
        for (uint32_t t = 0; t < 2; ++t) {
            for (uint32_t n = 0; n < DT; ++n) {
                matmul_tiles(cb_tiles, cb_res, T_DUP + t, n, t * DT + n);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_pv);
        for (uint32_t i = 0; i < 2 * DT; ++i) {
            pack_tile(i, cb_pv);
        }
        tile_regs_release();
        pv.push_back(2 * DT);
        res.pop_front(DT);

        // Column biases: w = max((m + ln den) / scale, NEG), b0 = trunc(w), b1 = trunc(w + ln(1 - e^((b0 - w) scale)) /
        // scale) with the log argument clamped at TINY.
        den.wait_front(1);
        bias.reserve_back(2);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_m);
        copy_init(cb_m);
        copy_tile(cb_m, 0, 0);
        copy_init(cb_den);
        copy_tile(cb_den, 0, 1);
        log_tile_init();
        log_tile(1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, INV_SCALE);
        unary_max_tile_init();
        unary_max_tile(0, NEG);
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(0, 1);
        MATH((trunc_bf16_tile(1)));
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(1, 2);
        sub_binary_tile_init();
        sub_binary_tile(2, 0, 2);
        binop_with_scalar_tile_init();
        mul_unary_tile(2, SCALE);
        exp_tile_init();
        exp_tile(2);
        binop_with_scalar_tile_init();
        rsub_unary_tile(2, ONE);
        unary_max_tile_init();
        unary_max_tile(2, TINY);
        log_tile_init();
        log_tile(2);
        binop_with_scalar_tile_init();
        mul_unary_tile(2, INV_SCALE);
        add_binary_tile_init();
        add_binary_tile(0, 2, 0);
        MATH((trunc_bf16_tile(0)));
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_bias);
        pack_tile(1, cb_bias);
        pack_tile(0, cb_bias);
        tile_regs_release();
        bias.push_back(2);
        m.pop_front(1);
        den.pop_front(1);
        q.pop_front(DT);
    }
}
