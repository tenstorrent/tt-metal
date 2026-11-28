// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Phase 2 compute of sdpa_decode_splitfree, the stats-first merge (GEMMA4_SDPA_SPLITFREE_COMBINE=stats): per
// combine job (bg, blk) of this core, over the n live units of the row (u = lo + i):
//   A. M = max_u m_u: the SFPU max in dest (exact and order-free), the running max in dest 0 of each window and
//      up to three more m tiles (every FPU copy of a window precedes its SFPU maxes);
//   B. w_u = exp((m_u - M) * scale) in column 0 (sub_exp_block's step), one CB_W page per unit;
//   C. L = sum_u w_u l_u, then per dest window of NB O tiles O_t = sum_u w_u O_u[t]: the FPU column-broadcast
//      multiply accumulates into the fp32 dest (mul_bcast_cols_init with acc_to_dest), u ascending;
//   D. O_t / L, one bf16 rounding.
// Every step is one fixed sequence over the row's units in ascending order and every O op is tile-local, so a
// tile's bits do not depend on CT, the job -> core split, the batch, the cores or the grid. Against the pairwise
// merge: no rescale chain of the accumulator (each O_u is scaled once, by its own w_u) and one multiply-accumulate
// per unit and O tile instead of ~4 packed ops.
//
// compile-time args: the sf:: prefix (CT_ACCESSORS args); defines EXP_APPROX_MODE, DHT_GRANULARITY (a divisor of
// CT), STATS_GRANULARITY, SUB_EXP_GRANULARITY. runtime args: [core]
#define LLK_ZEROFLAG_OUTLINE 1

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "splitfree_common.hpp"

void kernel_main() {
    using namespace sf;
    static_assert(STATS == 1, "the pairwise merge is compute_combine.cpp");
    const uint32_t core = get_arg_val<uint32_t>(0);
    constexpr uint16_t scale_bf16 = SCALE_BITS >> 16;
    constexpr uint32_t dst_window = 4;  // fp32 dest, half sync

    compute_kernel_hw_startup(CB_IN_S, CB_W, CB_ACC);

    CircularBuffer(CB_POS_C).wait_front(1);
    uint32_t pos[B];
    for (uint32_t b = 0; b < B; ++b) {
        pos[b] = read_tile_value(CB_POS_C, 0, b);
    }

    CircularBuffer cb_s(CB_IN_S);
    CircularBuffer cb_w(CB_W);
    CircularBuffer cb_m(CB_M);
    CircularBuffer cb_l(CB_L);
    CircularBuffer cb_o(CB_IN_O);
    CircularBuffer cb_acc(CB_ACC);
    for_each_combine_job(
        core, pos, [&](uint32_t /*b*/, uint32_t /*g*/, uint32_t /*bg*/, uint32_t /*blk*/, const Row& r) {
            const uint32_t n = unit_hi(r) - unit_lo(r) + 1;
            cb_s.wait_front(2 * NSLOTS);

            /* A: M = max_u m_u into CB_M */
            reconfig_data_format_srca(CB_IN_S);
            pack_reconfig_data_format(CB_M);
            copy_init(CB_IN_S);
            add_binary_tile_init();
            for (uint32_t i = 0; i < n;) {
                const bool first = (i == 0);
                const uint32_t take = first ? dst_window : dst_window - 1;
                const uint32_t i1 = (i + take < n) ? i + take : n;
                uint32_t d = 0;
                tile_regs_acquire();
                if (!first) {
                    cb_m.wait_front(1);
                    copy_tile(CB_M, 0, d++);
                }
                for (; i < i1; ++i) {
                    copy_tile(CB_IN_S, i, d++);
                }
                for (uint32_t k = 1; k < d; ++k) {
                    MATH((_llk_math_eltwise_binary_sfpu_params_(sdpa_max_sfpi, 0, k, 0, VectorMode::RC)));
                }
                tile_regs_commit();
                if (!first) {
                    cb_m.pop_front(1);
                }
                cb_m.reserve_back(1);
                tile_regs_wait();
                pack_tile(0, CB_M);
                tile_regs_release();
                cb_m.push_back(1);
            }

            /* B: w_u = exp((m_u - M) * scale), column 0, CB_W page u - lo */
            cb_m.wait_front(1);
            reconfig_data_format(CB_IN_S, CB_M);
            pack_reconfig_data_format(CB_W);
            sub_init(CB_IN_S, CB_M);
            exp_tile_init<EXP_APPROX_MODE>();
            cb_w.reserve_back(NSLOTS);
            for (uint32_t i = 0; i < n; ++i) {
                invalidate_l1_cache();
                tile_regs_acquire();
                sub_tiles(CB_IN_S, CB_M, i, 0, 0);
                MATH((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(0)));
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, CB_W, i);
                tile_regs_release();
            }
            cb_w.push_back(NSLOTS);
            cb_m.pop_front(1);

            /* C: L = sum_u w_u l_u, accumulated in dest */
            cb_w.wait_front(NSLOTS);
            reconfig_data_format(CB_IN_S, CB_W);
            pack_reconfig_data_format(CB_L);
            mul_bcast_cols_init(CB_IN_S, CB_W, true /*acc_to_dest*/);
            cb_l.reserve_back(1);
            tile_regs_acquire();
            for (uint32_t i = 0; i < n; ++i) {
                mul_tiles_bcast_cols(CB_IN_S, CB_W, NSLOTS + i, i, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_L);
            tile_regs_release();
            cb_l.push_back(1);
            cb_s.pop_front(2 * NSLOTS);

            /* C: O_t = sum_u w_u O_u[t], NB tiles per dest window, the units in read batches of UB */
            reconfig_data_format(CB_IN_O, CB_W);
            pack_reconfig_data_format(CB_ACC);
            mul_bcast_cols_init(CB_IN_O, CB_W, true /*acc_to_dest*/);
            for (uint32_t t0 = 0; t0 < CT; t0 += NB) {
                tile_regs_acquire();
                for (uint32_t i0 = 0; i0 < n; i0 += UB) {
                    const uint32_t i1 = (i0 + UB < n) ? i0 + UB : n;
                    cb_o.wait_front(UB * NB);
                    for (uint32_t i = i0; i < i1; ++i) {
                        for (uint32_t t = 0; t < NB; ++t) {
                            mul_tiles_bcast_cols(CB_IN_O, CB_W, (i - i0) * NB + t, i, t);
                        }
                    }
                    cb_o.pop_front(UB * NB);
                }
                tile_regs_commit();
                cb_acc.reserve_back(NB);
                tile_regs_wait();
                for (uint32_t t = 0; t < NB; ++t) {
                    pack_tile(t, CB_ACC);
                }
                tile_regs_release();
                cb_acc.push_back(NB);
            }
            cb_w.pop_front(NSLOTS);

            /* D: O / L, one bf16 rounding */
            recip_block_inplace(CB_L, 1);
            mul_block_bcast_cols<1, CT, false, false>(CB_ACC, CB_L, CB_OUT_FINAL);
        });
}
