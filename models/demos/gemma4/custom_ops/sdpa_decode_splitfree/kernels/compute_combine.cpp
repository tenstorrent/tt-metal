// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Phase 2 compute of sdpa_decode_splitfree: per (row, group) pair of this core, the live units' partials merged
// in ascending unit order with the stock tree-reduction step (sdpa_flash_decode.cpp's child merge on
// compute_common.hpp: correction_block gives m' = max(m, m_u), a = exp((m - m') scale), a_u = exp((m_u - m')
// scale), l' = l a + l_u a_u; then O' = O a + O_u a_u), every value fp32; then O / l once, packed to bf16.
// The order is fixed by the unit indices, so the result does not depend on the batch, the cores or the grid.
//
// compile-time args: the sf:: prefix (24); defines EXP_APPROX_MODE, DHT_GRANULARITY, STATS_GRANULARITY,
// SUB_EXP_GRANULARITY. runtime args: [core]
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
    constexpr uint32_t out_tiles = DHT;
    constexpr VectorMode vector_mode = VectorMode::RC;

    compute_kernel_hw_startup(CB_IN_M, CB_IN_L, CB_ACC);

    CircularBuffer(CB_POS_C).wait_front(1);
    uint32_t pos[B];
    for (uint32_t b = 0; b < B; ++b) {
        pos[b] = read_tile_value(CB_POS_C, 0, b);
    }

    for (uint32_t bg = core; bg < BG; bg += NCORES) {
        const Row r = row_of(pos[bg / G]);
        if (!r.live) {
            continue;
        }
        const uint32_t lo = unit_lo(r);
        const uint32_t hi = unit_hi(r);
        // every CB of the merge is fp32: set the pack format once per pair (the previous pair ended in bf16)
        pack_reconfig_data_format(CB_ACC);
        for (uint32_t u = lo; u <= hi; ++u) {
            if (u == lo) {
                /* acc <- the first unit */
                reconfig_data_format_srca(CB_IN_M);
                move_block<true>(CB_IN_M, CB_PREV_MAX, 1);
                move_block<true>(CB_IN_L, CB_PREV_SUM, 1);
                move_block<true>(CB_IN_O, CB_ACC, out_tiles);
                continue;
            }
            /* l_u -> PREV_SUM_2; the fused correction: CUR_MAX, CUR_SUM, EMD (for acc), EMD_2 (for the unit) */
            reconfig_data_format_srca(CB_IN_L);
            move_block<true>(CB_IN_L, CB_PREV_SUM_2, 1);
            correction_block<SCALE_BITS, vector_mode>(
                CB_IN_M, CB_PREV_SUM_2, CB_CUR_MAX, CB_PREV_MAX, CB_CUR_SUM, CB_PREV_SUM, CB_EMD, CB_EMD_2, 1);
            /* O = O * a + O_u * a_u */
            reconfig_data_format_srca(CB_IN_O);
            move_block<true>(CB_IN_O, CB_ACC_2, out_tiles);
            mul_block_bcast_cols_inplace<1, DHT>(CB_ACC, CB_EMD);
            mul_block_bcast_cols_inplace<1, DHT>(CB_ACC_2, CB_EMD_2);
            add_block_inplace<true>(CB_ACC, CB_ACC_2, out_tiles);
            /* prev <- cur */
            CircularBuffer(CB_PREV_MAX).pop_front(1);
            CircularBuffer(CB_IN_M).pop_front(1);
            reconfig_data_format_srca(CB_CUR_MAX);
            move_block<true>(CB_CUR_MAX, CB_PREV_MAX, 1);
            move_block<true>(CB_CUR_SUM, CB_PREV_SUM, 1);
        }
        /* O / l, one bf16 rounding */
        reconfig_data_format(CB_PREV_SUM, CB_PREV_SUM);
        pack_reconfig_data_format(CB_PREV_SUM);
        recip_block_inplace(CB_PREV_SUM, 1);
        reconfig_data_format(CB_ACC, CB_PREV_SUM);
        pack_reconfig_data_format(CB_ACC);
        mul_block_bcast_cols_inplace<1, DHT>(CB_ACC, CB_PREV_SUM);
        CircularBuffer(CB_PREV_MAX).pop_front(1);
        reconfig_data_format_srca(CB_ACC);
        pack_reconfig_data_format(CB_OUT_FINAL);
        move_block<true>(CB_ACC, CB_OUT_FINAL, out_tiles);
    }
}
