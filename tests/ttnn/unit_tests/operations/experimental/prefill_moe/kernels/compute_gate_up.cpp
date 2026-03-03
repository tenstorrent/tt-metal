// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Prefill MoE compute V0: Compute kernel (TRISC)
// Performs gate_up matmul: activation[1, K] × weights[K, N_per_core] → output[1, N_per_core]
// Accumulates across K dimension one tile at a time.
// Processes N_BLOCK output tiles per matmul_block call (ct_dim=N_BLOCK).
//
// CB0 (activation): 1 BF16 tile per K iteration
// CB1 (weights): N_per_core BFP4_b tiles per K iteration
// CB2 (output): N_per_core BF16 tiles (result)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"

void kernel_main() {
    constexpr uint32_t num_k_tiles = get_compile_time_arg_val(0);  // K dimension (90)
    constexpr uint32_t n_per_core = get_compile_time_arg_val(1);   // Output tiles per core (15)

    // Process N tiles in blocks of N_BLOCK for efficiency
    // N_PER_CORE=15, so try blocks of 5 (3 blocks per K iteration)
    constexpr uint32_t N_BLOCK = 5;
    constexpr uint32_t n_blocks = n_per_core / N_BLOCK;  // 3

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
