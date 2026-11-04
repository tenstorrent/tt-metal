// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Phase 1 compute of sdpa_decode_splitfree: for every unit of this core, the stock flash-decode recurrence over
// the unit's live chunks in ascending order (sdpa_flash_decode.cpp's per-chunk body on compute_common.hpp's
// helpers: QK^T, the causal / window mask adds, the running row max, exp((s - m) * scale), the row sum, P V,
// the lazy-softmax rescale of the accumulator) with the statistics and the accumulator in fp32 CBs, then the
// unit's (O, m, l) to the writer. Identical code over identical data in a fixed order on every core and
// grid: a unit's partial does not depend on who computes it.
//
// compile-time args: the sf:: prefix (24); defines EXP_APPROX_MODE, SUB_EXP_GRANULARITY, STATS_GRANULARITY,
// DHT_GRANULARITY as the stock program factory sets them. runtime args: [core]
#define LLK_ZEROFLAG_OUTLINE 1

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
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
    const uint32_t core = get_arg_val<uint32_t>(0);
    constexpr uint32_t q_tiles = DHT;                   // one Q tile row
    constexpr uint32_t qk_tiles = SK;                   // one row of scores
    constexpr uint32_t out_tiles = DHT;                 // one row of O
    constexpr VectorMode vector_mode = VectorMode::RC;  // full 32x32 tiles

    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_Q, CB_K, CB_QK);
    matmul_init(CB_Q, CB_K);

    // cur_pos, read through the unpack thread's mailbox so the three threads agree on every branch
    CircularBuffer(CB_POS_C).wait_front(1);
    uint32_t pos[B];
    for (uint32_t b = 0; b < B; ++b) {
        pos[b] = read_tile_value(CB_POS_C, 0, b);
    }

    for_each_unit(core, pos, [&](uint32_t /*b*/, uint32_t /*g*/, uint32_t /*bg*/, uint32_t u, const Row& r) {
        CircularBuffer(CB_Q).wait_front(q_tiles);
        const uint32_t j_lo = chunk_lo(r, u);
        const uint32_t j_hi = chunk_hi(r, u);
        uint32_t cb_out_mm = CB_OUT_ACC;  // the first chunk's PV lands in the accumulator, later ones in OUT_IM
        for (uint32_t j = j_lo; j <= j_hi; ++j) {
            /* S = Q @ K^T */
            reconfig_data_format(CB_K, CB_Q);
            pack_reconfig_data_format(CB_QK);
            matmul_blocks(
                CB_Q, CB_K, CB_QK, 1, SK, DHT, 1, QK_IN1_SUB, DHT, 1, QK_SUB_W, true, false, CB_MASK, CB_ZERO);

            /* S += mask (the chunk holding cur_pos; the first live chunk of a window) */
            if (chunk_has_tail_mask(r, j)) {
                reconfig_data_format(CB_QK, CB_MASK);
                add_block_inplace<true>(CB_QK, CB_MASK, qk_tiles);
            }
            if (chunk_has_window_mask(r, j)) {
                reconfig_data_format(CB_QK, CB_WMASK);
                add_block_inplace<true>(CB_QK, CB_WMASK, qk_tiles);
            }

            /* m = max(prev_m, rowmax(S)) */
            reconfig_data_format(CB_QK, CB_SCALE);
            pack_reconfig_data_format(CB_CUR_MAX);
            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, CB_QK, CB_SCALE, 1, vector_mode>(
                CB_CUR_MAX, CB_PREV_MAX, SK, j > j_lo);

            /* S = exp((S - m) * scale) */
            reconfig_data_format(CB_QK, CB_CUR_MAX);
            pack_reconfig_data_format(CB_QK);
            sub_exp_block_bcast_cols_inplace<CB_QK, 1, SCALE_BITS, true, false, vector_mode>(
                CB_CUR_MAX, CB_CUR_SUM, SK);
            CircularBuffer(CB_QK).wait_front(qk_tiles);

            /* l_chunk = rowsum(S) */
            reconfig_data_format(CB_QK, CB_SCALE);
            pack_reconfig_data_format(CB_CUR_SUM);
            reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, CB_QK, CB_SCALE, 1, vector_mode>(
                CB_CUR_SUM, CB_CUR_SUM, SK, false);

            /* O_chunk = S @ V */
            reconfig_data_format(CB_V, CB_QK);
            pack_reconfig_data_format(CB_OUT_IM);
            matmul_blocks(
                CB_QK, CB_V, cb_out_mm, 1, DHT, SK, 1, OUT_IN1_SUB, SK, 1, OUT_SUB_W, false, false, CB_MASK, CB_ZERO);
            reconfig_data_format_srca(CB_OUT_IM);
            CircularBuffer(CB_QK).pop_front(qk_tiles);

            if (j == j_lo) {
                cb_out_mm = CB_OUT_IM;
            } else {
                /* a = exp((prev_m - m) * scale); l = l * a + l_chunk; O = O * a + O_chunk */
                reconfig_data_format(CB_PREV_MAX, CB_CUR_MAX);
                pack_reconfig_data_format(CB_EMD);
                sub_exp_block<SCALE_BITS>(CB_PREV_MAX, CB_CUR_MAX, CB_EMD, 1);
                CircularBuffer(CB_PREV_MAX).pop_front(1);
                mul_block_inplace(CB_PREV_SUM, CB_EMD, 1);
                reconfig_data_format(CB_OUT_ACC, CB_EMD);
                pack_reconfig_data_format(CB_OUT_ACC);
                mul_block_bcast_cols<1, DHT, true, false>(CB_OUT_ACC, CB_EMD, CB_OUT_ACC);
                reconfig_data_format(CB_CUR_SUM, CB_PREV_SUM);
                pack_reconfig_data_format(CB_CUR_SUM);
                add_block_inplace<true>(CB_CUR_SUM, CB_PREV_SUM, 1);
                reconfig_data_format(CB_OUT_ACC, CB_OUT_IM);
                pack_reconfig_data_format(CB_OUT_ACC);
                add_block_inplace<true>(CB_OUT_ACC, CB_OUT_IM, out_tiles);
            }

            /* prev <- cur */
            reconfig_data_format(CB_CUR_MAX, CB_CUR_MAX);
            pack_reconfig_data_format(CB_PREV_MAX);
            move_block<true>(CB_CUR_MAX, CB_PREV_MAX, 1);
            move_block<true>(CB_CUR_SUM, CB_PREV_SUM, 1);
        }

        /* the unit's partial (O, m, l) to the writer */
        reconfig_data_format_srca(CB_OUT_ACC);
        pack_reconfig_data_format(CB_OUT_O);
        move_block<true>(CB_OUT_ACC, CB_OUT_O, out_tiles);
        reconfig_data_format_srca(CB_PREV_MAX);
        pack_reconfig_data_format(CB_OUT_M);
        move_block<true>(CB_PREV_MAX, CB_OUT_M, 1);
        reconfig_data_format_srca(CB_PREV_SUM);
        pack_reconfig_data_format(CB_OUT_L);
        move_block<true>(CB_PREV_SUM, CB_OUT_L, 1);
        CircularBuffer(CB_Q).pop_front(q_tiles);
    });
}
