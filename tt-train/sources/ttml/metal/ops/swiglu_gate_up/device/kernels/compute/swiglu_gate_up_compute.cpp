// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

// ============================================================================
// SwiGLU Gate-Up Compute Kernel (Design A Step 1)
//
// Computes M = SiLU(X @ W1) * (X @ W3) using 2D multicast matmul tiling.
//
// Each core (r, c) in the R×C grid computes its block:
//   XW1[per_core_M, per_core_N] = X[per_core_M, K] @ W1[K, per_core_N]
//   XW3[per_core_M, per_core_N] = X[per_core_M, K] @ W3[K, per_core_N]
//   M[per_core_M, per_core_N] = SiLU(XW1) * XW3
//
// Loop structure (per M tile-row):
//   for k_block in K_blocks:
//     wait X[K_block_size]
//     for n_sub in N_blocks:
//       wait W1[K_block_size × block_size], matmul → acc XW1[n_sub]
//       wait W3[K_block_size × block_size], matmul → acc XW3[n_sub]
//     pop X
//   SiLU(XW1) * XW3 → M
// ============================================================================

constexpr uint32_t per_core_M = get_compile_time_arg_val(0);
constexpr uint32_t per_core_N = get_compile_time_arg_val(1);
constexpr uint32_t per_core_N_rounded = get_compile_time_arg_val(2);
constexpr uint32_t block_size = get_compile_time_arg_val(3);
constexpr uint32_t Wt = get_compile_time_arg_val(4);
constexpr uint32_t num_n_blocks = get_compile_time_arg_val(5);

constexpr uint32_t tiles_per_batch = block_size * block_size;

constexpr auto cb_in0_idx = tt::CBIndex::c_0;
constexpr auto cb_w1_idx = tt::CBIndex::c_1;
constexpr auto cb_w3_idx = tt::CBIndex::c_2;
constexpr auto cb_xw1_acc_idx = tt::CBIndex::c_3;
constexpr auto cb_xw3_acc_idx = tt::CBIndex::c_4;
constexpr auto cb_m_out_idx = tt::CBIndex::c_5;

// Accumulate X @ W into the accumulation CB using packer L1 accumulation.
inline void matmul_accumulate_l1(
    const tt::CBIndex cb_x_idx,
    const tt::CBIndex cb_w_idx,
    const tt::CBIndex cb_acc_idx,
    const uint32_t n_block_offset,
    const uint32_t k_block_size,
    const bool first_k_block) {
    tile_regs_acquire();

    cb_wait_front(cb_w_idx, tiles_per_batch);

    mm_block_init_short(
        cb_x_idx, cb_w_idx, /*transpose=*/false, /*ct_dim=*/block_size, /*rt_dim=*/1U, /*kt_dim=*/k_block_size);

    uint32_t in0_index = 0U;
    uint32_t in1_index = 0U;
    for (uint32_t k = 0U; k < k_block_size; ++k) {
        matmul_block(
            cb_x_idx,
            cb_w_idx,
            in0_index,
            in1_index,
            /*dst_index=*/0U,
            /*transpose=*/false,
            /*ct_dim=*/block_size,
            /*rt_dim=*/1U,
            /*kt_dim=*/k_block_size);
        in0_index++;
        in1_index += block_size;
    }

    cb_pop_front(cb_w_idx, tiles_per_batch);

    tile_regs_commit();
    pack_l1_acc_block(cb_acc_idx, first_k_block, block_size, n_block_offset);
}

// Process a single SiLU tile: M = SiLU(XW1) * XW3 = (XW1 * sigmoid(XW1)) * XW3
// Uses 3 consecutive DST registers starting at base_reg: [xw1, xw3, scratch]
inline void compute_silu_tile(uint32_t tile_offset, uint32_t base_reg) {
    copy_tile_init(cb_xw1_acc_idx);
    copy_tile(cb_xw1_acc_idx, tile_offset, base_reg);
    copy_tile_init(cb_xw3_acc_idx);
    copy_tile(cb_xw3_acc_idx, tile_offset, base_reg + 1U);

    copy_dest_values_init();
    copy_dest_values(base_reg, base_reg + 2U);  // scratch = XW1
    sigmoid_tile_init();
    sigmoid_tile(base_reg + 2U);  // scratch = sigmoid(XW1)
    mul_binary_tile_init();
    mul_binary_tile(base_reg, base_reg + 2U, base_reg);  // base = XW1 * sigmoid = SiLU
    mul_binary_tile(base_reg, base_reg + 1U, base_reg);  // base = SiLU * XW3 = M
}

void kernel_main() {
    init_sfpu(cb_in0_idx, cb_m_out_idx);
    binary_op_init_common(cb_in0_idx, cb_w1_idx, cb_m_out_idx);

    for (uint32_t m = 0U; m < per_core_M; ++m) {
        // ---- Accumulate XW1 and XW3 for this M tile-row ----
        cb_reserve_back(cb_xw1_acc_idx, per_core_N_rounded);
        cb_reserve_back(cb_xw3_acc_idx, per_core_N_rounded);

        for (uint32_t k_block_start = 0U; k_block_start < Wt; k_block_start += block_size) {
            const uint32_t k_block_size = std::min(block_size, Wt - k_block_start);
            const bool first_k_block = (k_block_start == 0U);

            cb_wait_front(cb_in0_idx, block_size);

            for (uint32_t n_sub = 0U; n_sub < num_n_blocks; ++n_sub) {
                uint32_t n_offset = n_sub * block_size;

                matmul_accumulate_l1(cb_in0_idx, cb_w1_idx, cb_xw1_acc_idx, n_offset, k_block_size, first_k_block);

                matmul_accumulate_l1(cb_in0_idx, cb_w3_idx, cb_xw3_acc_idx, n_offset, k_block_size, first_k_block);
            }

            cb_pop_front(cb_in0_idx, block_size);
        }

        cb_push_back(cb_xw1_acc_idx, per_core_N_rounded);
        cb_push_back(cb_xw3_acc_idx, per_core_N_rounded);

        // ---- SiLU fusion: M = SiLU(XW1) * XW3 ----
        cb_wait_front(cb_xw1_acc_idx, per_core_N_rounded);
        cb_wait_front(cb_xw3_acc_idx, per_core_N_rounded);

        for (uint32_t n_block_start = 0U; n_block_start < per_core_N; n_block_start += block_size) {
            const uint32_t n_block_size = std::min(block_size, per_core_N - n_block_start);

            uint32_t n = 0U;
            // Batched: process 2 tiles per acquire/commit when possible
            for (; n + 1U < n_block_size; n += 2U) {
                tile_regs_acquire();
                compute_silu_tile(n_block_start + n, 0U);
                compute_silu_tile(n_block_start + n + 1U, 1U);
                tile_regs_commit();
                pack_and_push_block(cb_m_out_idx, 2U);
            }
            if (n < n_block_size) {
                tile_regs_acquire();
                compute_silu_tile(n_block_start + n, 0U);
                tile_regs_commit();
                pack_and_push(0U, cb_m_out_idx);
            }

            // Pad to block_size if needed
            if (n_block_size < block_size) {
                tile_regs_acquire();
                tile_regs_commit();
                pack_and_push_block(cb_m_out_idx, block_size - n_block_size);
            }
        }

        cb_pop_front(cb_xw1_acc_idx, per_core_N_rounded);
        cb_pop_front(cb_xw3_acc_idx, per_core_N_rounded);
    }
}
