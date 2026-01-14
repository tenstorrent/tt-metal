// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

template <uint32_t CbA, uint32_t CbB, uint32_t CbOut, uint32_t NumTilesK, uint32_t OutputTileId = 0>
FORCE_INLINE void matmul_with_relu_block() {
    cb_wait_front(CbA, NumTilesK);
    cb_wait_front(CbB, NumTilesK);
    cb_reserve_back(CbOut, NumTilesK);

    tile_regs_acquire();

    for (uint32_t k = 0; k < NumTilesK; k++) {
        matmul_tiles(CbA, CbB, k, k, 0);
    }
    relu_tile(0);

    tile_regs_commit();

    // cb_pop_front(CbA, NumTiles); // Don't pop here because we need to use the input again for next stage
    cb_pop_front(CbB, NumTilesK);

    tile_regs_wait();
    pack_tile(0, CbOut, OutputTileId);  // Pack at offset OutputTileId
    tile_regs_release();

    cb_push_back(CbOut, NumTilesK);
}

constexpr uint32_t MATMUL_ACC_REG_ID = 0;
constexpr uint32_t BIAS_REG_ID = 1;

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbBias,
    uint32_t CbOut,
    uint32_t NumTilesK,
    uint32_t NumTilesBias,
    uint32_t OutputTileId = 0>
FORCE_INLINE void matmul_with_bias_block() {
    cb_wait_front(CbA, NumTilesK);
    cb_wait_front(CbB, NumTilesK);
    cb_wait_front(CbBias, NumTilesBias);
    cb_reserve_back(CbOut, NumTilesK);

    tile_regs_acquire();

    init_sfpu(CbA, CbOut);  // Hangs if we put this at the beginning of the program
    mm_init_short(CbA, CbB);
    for (uint32_t k = 0; k < NumTilesK; k++) {
        matmul_tiles(CbA, CbB, k, k, MATMUL_ACC_REG_ID);
    }
    copy_tile_init(CbBias);  // Hangs if we put this at the beginning of the program
    copy_tile(CbBias, 0, BIAS_REG_ID);

    add_binary_tile_init();
    add_binary_tile(MATMUL_ACC_REG_ID, BIAS_REG_ID, MATMUL_ACC_REG_ID);  // Accumulate the bias into the matmul result

    tile_regs_commit();

    cb_pop_front(CbA, NumTilesK);
    cb_pop_front(CbB, NumTilesK);
    cb_pop_front(CbBias, NumTilesBias);

    tile_regs_wait();
    pack_tile(MATMUL_ACC_REG_ID, CbOut, OutputTileId);  // Pack at offset OutputTileId
    tile_regs_release();

    cb_push_back(CbOut, NumTilesK);
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t in0_cb = get_compile_time_arg_val(0);
    constexpr uint32_t weight0_cb = get_compile_time_arg_val(1);
    constexpr uint32_t weight1_cb = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb = get_compile_time_arg_val(3);
    constexpr uint32_t interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t interm_cb2 = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(6);
    constexpr bool fp32_dest_acc_en = get_compile_time_arg_val(7);

    constexpr uint32_t num_output_tiles = 1;
    constexpr uint32_t out_subblock_h = 1;
    constexpr uint32_t out_subblock_w = 1;
    constexpr uint32_t in0_block_w = 1;  // Process one K tile at a time
    mm_block_init(in0_cb, weight0_cb, interm_cb, false, out_subblock_w, out_subblock_h, in0_block_w);
    relu_tile_init();

    matmul_with_relu_block<in0_cb, weight0_cb, interm_cb, num_tiles_k>();
    matmul_with_bias_block<interm_cb2, weight1_cb, in0_cb, out_cb, num_tiles_k, num_output_tiles>();
}
}  // namespace NAMESPACE
