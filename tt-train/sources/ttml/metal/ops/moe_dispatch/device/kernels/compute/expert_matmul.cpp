// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// MoE Dispatch — Expert Matmul Compute
//
// Identical to the proven moe_expert_fwd compute kernel (PCC 0.9999).
// For each token tile-row received from the socket:
//   Y[row] = X[row] @ W_expert   using matmul_block + L1 accumulation
//
// The receiver reader feeds input tiles (cb_in) and weight tiles (cb_w)
// in matmul block order. This kernel just consumes them.
// ============================================================================

#include <algorithm>
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t K_t = get_compile_time_arg_val(0);
constexpr uint32_t N_t = get_compile_time_arg_val(1);
constexpr uint32_t block_size = get_compile_time_arg_val(2);
constexpr uint32_t num_experts_local = get_compile_time_arg_val(3);

constexpr auto cb_in_idx = tt::CBIndex::c_0;
constexpr auto cb_w_idx = tt::CBIndex::c_1;
constexpr auto cb_out_idx = tt::CBIndex::c_10;

constexpr uint32_t tiles_per_batch = block_size * block_size;
constexpr uint32_t N_t_rounded = ((N_t + block_size - 1) / block_size) * block_size;

inline void compute_row() {
    cb_reserve_back(cb_out_idx, N_t_rounded);

    for (uint32_t p_start = 0U; p_start < K_t; p_start += block_size) {
        const uint32_t p_size = std::min(block_size, K_t - p_start);
        const bool first_p = (p_start == 0U);

        cb_wait_front(cb_in_idx, block_size);

        for (uint32_t n_start = 0U; n_start < N_t; n_start += block_size) {
            tile_regs_acquire();
            cb_wait_front(cb_w_idx, tiles_per_batch);

            if (n_start == 0U) {
                mm_block_init_short(
                    cb_in_idx,
                    cb_w_idx,
                    /*transpose=*/false,
                    /*ct_dim=*/block_size,
                    /*rt_dim=*/1U,
                    /*kt_dim=*/p_size);
            }

            for (uint32_t p = 0U; p < p_size; ++p) {
                matmul_block(
                    cb_in_idx,
                    cb_w_idx,
                    /*in0_index=*/p,
                    /*in1_index=*/p * block_size,
                    /*dst_index=*/0U,
                    /*transpose=*/false,
                    /*ct_dim=*/block_size,
                    /*rt_dim=*/1U,
                    /*kt_dim=*/p_size);
            }

            cb_pop_front(cb_w_idx, tiles_per_batch);
            tile_regs_commit();
            pack_l1_acc_block(cb_out_idx, first_p, block_size, n_start);
        }

        cb_pop_front(cb_in_idx, block_size);
    }

    cb_push_back(cb_out_idx, N_t_rounded);
}

void kernel_main() {
    mm_init(cb_in_idx, cb_w_idx, cb_out_idx);

    for (uint32_t e = 0U; e < num_experts_local; ++e) {
        const uint32_t n_rows = get_arg_val<uint32_t>(e);
        for (uint32_t r = 0U; r < n_rows; ++r) {
            compute_row();
        }
    }
}
