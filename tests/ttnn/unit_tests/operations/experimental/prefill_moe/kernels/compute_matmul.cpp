// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Generic matmul compute kernel with configurable N_BLOCK.
// Performs: activation[1, K] × weights[K, N_per_core] → output[1, N_per_core]
// Accumulates across K dimension one tile at a time.
//
// CB0 (activation): 1 BF16 tile per K iteration
// CB1 (weights): N_per_core tiles per K iteration (any format)
// CB2 (output): N_per_core BF16 tiles (result)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"

void kernel_main() {
    constexpr uint32_t num_k_tiles = get_compile_time_arg_val(0);  // K dimension in tiles
    constexpr uint32_t n_per_core = get_compile_time_arg_val(1);   // Output tiles per core
    constexpr uint32_t N_BLOCK = get_compile_time_arg_val(2);      // Tiles per matmul_block (must be ≤7)
    constexpr uint32_t n_blocks = n_per_core / N_BLOCK;

    constexpr uint32_t cb_act = 0;
    constexpr uint32_t cb_weights = 1;
    constexpr uint32_t cb_out = 2;

    mm_block_init(cb_act, cb_weights, cb_out, /*transpose=*/false, /*ct_dim=*/N_BLOCK, /*rt_dim=*/1, /*kt_dim=*/1);

    tile_regs_acquire();

    for (uint32_t k = 0; k < num_k_tiles; ++k) {
        cb_wait_front(cb_act, 1);
        cb_wait_front(cb_weights, n_per_core);

        for (uint32_t b = 0; b < n_blocks; ++b) {
            uint32_t n_offset = b * N_BLOCK;
            matmul_block(
                cb_act,
                cb_weights,
                /*in0_index=*/0,
                /*in1_index=*/n_offset,
                /*idst=*/n_offset,
                /*transpose=*/false,
                /*ct_dim=*/N_BLOCK,
                /*rt_dim=*/1,
                /*kt_dim=*/1);
        }

        cb_pop_front(cb_act, 1);
        cb_pop_front(cb_weights, n_per_core);
    }

    tile_regs_commit();
    tile_regs_wait();

    cb_reserve_back(cb_out, n_per_core);
    for (uint32_t i = 0; i < n_per_core; ++i) {
        pack_tile(i, cb_out);
    }
    cb_push_back(cb_out, n_per_core);

    tile_regs_release();
}
