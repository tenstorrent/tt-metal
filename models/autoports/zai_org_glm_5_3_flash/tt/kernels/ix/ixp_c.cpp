// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Indexer phase member compute (ixp_common.hpp), v2. Per CB_K block (KC key tiles):
//   phase 1, every resident query r: S_r = relu(Q_r @ K^T) (one acquire: 4 K-step matmul_blocks of 1 x KC tiles,
//            32 heads x 32 pools per tile), packed with the packer's ReLU as bf16 into CB_SR at r KC + j;
//   phase 2, ONE acquire: sum over r of WS_r @ S_r accumulated in DST (row r of WS_r = query r's head weights, the
//            other rows zero, so the sum keeps every query's row) -> the block's KC fp32 score tiles in CB_SC and
//            again in CB_SC2 (the two filter RISCs' copies).
// Two mode switches per block (the v1 form paid two per query, an fp32 S round trip and an L1-accumulated pack).
// Half-sync fp32 DST, KC <= 4. Compile-time: NQ, KC. Runtime: blocks.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/ix/ixp_common.hpp"

using namespace ixp;

void kernel_main() {
    constexpr uint32_t NQ = get_compile_time_arg_val(0);
    constexpr uint32_t KC = get_compile_time_arg_val(1);
    const uint32_t nblk = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_Q, CB_K, CB_SR);
    cb_wait_front(CB_Q, 4 * NQ);
    cb_wait_front(CB_WS, NQ);
    cb_wait_front(CB_ONE, 1);
    for (uint32_t b = 0; b < nblk; ++b) {
        cb_wait_front(CB_K, 4 * KC);
        cb_reserve_back(CB_SR, NQ * KC);
        reconfig_data_format(CB_K, CB_Q);
        matmul_block_init(CB_Q, CB_K, true, KC, 1, 4);
        pack_reconfig_data_format(CB_SR);
        pack_relu_config(ReluConfig::zero());
        for (uint32_t r = 0; r < NQ; ++r) {
#if defined(IXP_DIAG) && (IXP_DIAG & 2)
            break;  // diagnostic: no math (the CBs still move)
#endif
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < 4; ++ki) {
                matmul_block(CB_Q, CB_K, 4 * r + ki, ki * KC, 0, true, KC, 1, 4);  // K-step ki: tiles ki KC ..
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < KC; ++j) {
                pack_tile<true>(j, CB_SR, r * KC + j);
            }
            tile_regs_release();
        }
        pack_relu_config(ReluConfig::none());
        cb_push_back(CB_SR, NQ * KC);
        cb_pop_front(CB_K, 4 * KC);  // the slot goes back to the streamer before the weight pass

        cb_wait_front(CB_SR, NQ * KC);
        cb_reserve_back(CB_SC, 2 * KC);
        cb_reserve_back(CB_SC2, 2 * KC);
        cb_reserve_back(CB_SM, KC);
        reconfig_data_format(CB_SR, CB_WS);
        matmul_block_init(CB_WS, CB_SR, false, KC, 1, 1);
        tile_regs_acquire();
#if !(defined(IXP_DIAG) && (IXP_DIAG & 2))
        for (uint32_t r = 0; r < NQ; ++r) {
            matmul_block(CB_WS, CB_SR, r, r * KC, 0, false, KC, 1, 1);
        }
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_SC);
        for (uint32_t j = 0; j < KC; ++j) {
            pack_tile<true>(j, CB_SC, j);
        }
        for (uint32_t j = 0; j < KC; ++j) {  // the same scores for the BRISC filter's rows
            pack_tile<true>(j, CB_SC2, j);
        }
        for (uint32_t j = 0; j < KC; ++j) {  // and for the row maxima below
            pack_tile(j, CB_SM);
        }
        tile_regs_release();
        cb_push_back(CB_SM, KC);
        cb_pop_front(CB_SR, NQ * KC);
        // v8: row r's max over each key tile's 32 pools -> column 0 of tile KC + j of both score blocks
        cb_wait_front(CB_SM, KC);
        reconfig_data_format(CB_SM, CB_ONE);
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(CB_SM, CB_ONE, CB_SC);
        tile_regs_acquire();
        for (uint32_t j = 0; j < KC; ++j) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(CB_SM, CB_ONE, j, 0, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < KC; ++j) {
            pack_tile<true>(j, CB_SC, KC + j);
        }
        for (uint32_t j = 0; j < KC; ++j) {
            pack_tile<true>(j, CB_SC2, KC + j);
        }
        tile_regs_release();
        reduce_uninit();
        cb_pop_front(CB_SM, KC);
        cb_push_back(CB_SC, 2 * KC);
        cb_push_back(CB_SC2, 2 * KC);
    }
    cb_pop_front(CB_Q, 4 * NQ);
    cb_pop_front(CB_WS, NQ);
    cb_pop_front(CB_ONE, 1);
}
