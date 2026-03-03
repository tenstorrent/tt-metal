// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Prefill MoE compute: gate_up matmul + SwiGLU
// Performs gate_up matmul: activation[1, K] × weights[K, N_per_core] → [1, N_per_core]
// Weights are pre-shuffled so each core's N_per_core tiles = [N_per_core/2 gate, N_per_core/2 up]
// Then applies SwiGLU on dest registers (PACK thread):
//   result[j] = (clamp(up[j], -7, 7) + 1) * clamp(gate[j], max=7) * sigmoid(1.702 * clamp(gate[j], max=7))
// Output is N_per_core/2 tiles (the SwiGLU result).
//
// CB0 (activation): 1 BF16 tile per K iteration
// CB1 (weights): N_per_core BFP4_b tiles per K iteration
// CB2 (output): N_per_core/2 BF16 tiles (SwiGLU result)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"

#ifdef TRISC_PACK
#include "swiglu_sfpu.h"
#endif

void kernel_main() {
    constexpr uint32_t num_k_tiles = get_compile_time_arg_val(0);  // K dimension (90)
    constexpr uint32_t n_per_core = get_compile_time_arg_val(1);   // Weight tiles per core (12)
    constexpr uint32_t n_swiglu = n_per_core / 2;                  // SwiGLU output tiles (6)

    // Process N tiles in blocks of N_BLOCK (must be ≤7 for half-sync mode)
    constexpr uint32_t N_BLOCK = 6;
    constexpr uint32_t n_blocks = n_per_core / N_BLOCK;  // 2

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

    // SwiGLU on dest registers (PACK thread only)
    // dest[0..n_swiglu-1] = gate tiles, dest[n_swiglu..n_per_core-1] = up tiles
    // Pairs gate[j] with up[j + n_swiglu], output overwrites gate[j] in dest
#ifdef TRISC_PACK
    ckernel::llk_math_eltwise_binary_sfpu_swiglu_init<true>();
    for (uint32_t j = 0; j < n_swiglu; ++j) {
        ckernel::llk_math_eltwise_binary_sfpu_swiglu<true, false>(j, j + n_swiglu, j);
    }
#endif

    // Pack only the SwiGLU output tiles (first n_swiglu tiles in dest)
    cb_reserve_back(cb_out, n_swiglu);
    for (uint32_t i = 0; i < n_swiglu; ++i) {
        pack_tile(i, cb_out);
    }
    cb_push_back(cb_out, n_swiglu);

    tile_regs_release();
}
