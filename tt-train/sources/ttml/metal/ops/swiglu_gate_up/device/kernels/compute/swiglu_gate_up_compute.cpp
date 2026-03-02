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
// SwiGLU Gate-Up Compute Kernel (Design A Step 1) — with M sub-blocking
//
// Outer loop iterates over M-blocks of block_h rows. For each M-block, the
// full K-loop runs once, reusing weight tiles across all block_h M rows.
// This amortizes weight DRAM reads by block_h×.
//
// Performance-critical: mm_block_init_short and pack_reconfig_* are hoisted
// out of the per-row loop to avoid redundant hardware reconfiguration.
// ============================================================================

constexpr uint32_t per_core_N = get_compile_time_arg_val(0);
constexpr uint32_t per_core_N_rounded = get_compile_time_arg_val(1);
constexpr uint32_t block_size = get_compile_time_arg_val(2);
constexpr uint32_t Wt = get_compile_time_arg_val(3);
constexpr uint32_t num_n_blocks = get_compile_time_arg_val(4);
constexpr uint32_t block_h = get_compile_time_arg_val(5);
constexpr uint32_t num_m_blocks = get_compile_time_arg_val(6);

constexpr uint32_t tiles_per_batch = block_size * block_size;
constexpr uint32_t x_tiles_per_block = block_h * block_size;
constexpr uint32_t acc_tiles_total = block_h * per_core_N_rounded;

constexpr auto cb_in0_idx = tt::CBIndex::c_0;
constexpr auto cb_w1_idx = tt::CBIndex::c_1;
constexpr auto cb_w3_idx = tt::CBIndex::c_2;
constexpr auto cb_xw1_acc_idx = tt::CBIndex::c_3;
constexpr auto cb_xw3_acc_idx = tt::CBIndex::c_4;
constexpr auto cb_m_out_idx = tt::CBIndex::c_5;

// Matmul one row and pack to L1 accumulator — no init_short, no pack reconfig.
inline void matmul_one_row_fast(
    const tt::CBIndex cb_x_idx,
    const tt::CBIndex cb_w_idx,
    const tt::CBIndex cb_acc_idx,
    const uint32_t x_row_offset,
    const uint32_t acc_offset,
    const uint32_t k_block_size) {
    tile_regs_acquire();

    uint32_t in0_index = x_row_offset;
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

    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0U; t < block_size; ++t) {
        pack_tile</* out_of_order_output = */ true>(t, cb_acc_idx, acc_offset + t);
    }
    tile_regs_release();
}

inline void compute_silu_tile(uint32_t tile_offset, uint32_t base_reg) {
    copy_tile_init(cb_xw1_acc_idx);
    copy_tile(cb_xw1_acc_idx, tile_offset, base_reg);
    copy_tile_init(cb_xw3_acc_idx);
    copy_tile(cb_xw3_acc_idx, tile_offset, base_reg + 1U);

    copy_dest_values_init();
    copy_dest_values(base_reg, base_reg + 2U);
    sigmoid_tile_init();
    sigmoid_tile(base_reg + 2U);
    mul_binary_tile_init();
    mul_binary_tile(base_reg, base_reg + 2U, base_reg);
    mul_binary_tile(base_reg, base_reg + 1U, base_reg);
}

// Process block_h rows of matmul for one weight matrix (W1 or W3).
// Hoists mm_block_init_short and pack_reconfig outside the row loop.
inline void matmul_rows_for_weight(
    const tt::CBIndex cb_w_idx,
    const tt::CBIndex cb_acc_idx,
    const uint32_t n_offset,
    const uint32_t k_block_size,
    const bool first_k_block) {
    mm_block_init_short(
        cb_in0_idx, cb_w_idx, /*transpose=*/false, /*ct_dim=*/block_size, /*rt_dim=*/1U, /*kt_dim=*/k_block_size);
    pack_reconfig_data_format(cb_acc_idx);
    pack_reconfig_l1_acc(first_k_block ? 0 : 1U);

    for (uint32_t m_sub = 0U; m_sub < block_h; ++m_sub) {
        matmul_one_row_fast(
            cb_in0_idx, cb_w_idx, cb_acc_idx, m_sub * block_size, m_sub * per_core_N_rounded + n_offset, k_block_size);
    }

    pack_reconfig_l1_acc(0);
}

void kernel_main() {
    init_sfpu(cb_in0_idx, cb_m_out_idx);
    binary_op_init_common(cb_in0_idx, cb_w1_idx, cb_m_out_idx);

    for (uint32_t mb = 0U; mb < num_m_blocks; ++mb) {
        cb_reserve_back(cb_xw1_acc_idx, acc_tiles_total);
        cb_reserve_back(cb_xw3_acc_idx, acc_tiles_total);

        for (uint32_t k_block_start = 0U; k_block_start < Wt; k_block_start += block_size) {
            const uint32_t k_block_size = std::min(block_size, Wt - k_block_start);
            const bool first_k_block = (k_block_start == 0U);

            cb_wait_front(cb_in0_idx, x_tiles_per_block);

            for (uint32_t n_sub = 0U; n_sub < num_n_blocks; ++n_sub) {
                const uint32_t n_offset = n_sub * block_size;

                cb_wait_front(cb_w1_idx, tiles_per_batch);
                matmul_rows_for_weight(cb_w1_idx, cb_xw1_acc_idx, n_offset, k_block_size, first_k_block);
                cb_pop_front(cb_w1_idx, tiles_per_batch);

                cb_wait_front(cb_w3_idx, tiles_per_batch);
                matmul_rows_for_weight(cb_w3_idx, cb_xw3_acc_idx, n_offset, k_block_size, first_k_block);
                cb_pop_front(cb_w3_idx, tiles_per_batch);
            }

            cb_pop_front(cb_in0_idx, x_tiles_per_block);
        }

        cb_push_back(cb_xw1_acc_idx, acc_tiles_total);
        cb_push_back(cb_xw3_acc_idx, acc_tiles_total);

        // ---- SiLU fusion: M = SiLU(XW1) * XW3 for all block_h rows ----
        cb_wait_front(cb_xw1_acc_idx, acc_tiles_total);
        cb_wait_front(cb_xw3_acc_idx, acc_tiles_total);

        for (uint32_t m_sub = 0U; m_sub < block_h; ++m_sub) {
            const uint32_t row_offset = m_sub * per_core_N_rounded;

            for (uint32_t n_block_start = 0U; n_block_start < per_core_N; n_block_start += block_size) {
                const uint32_t n_block_size = std::min(block_size, per_core_N - n_block_start);

                uint32_t n = 0U;
                for (; n + 1U < n_block_size; n += 2U) {
                    tile_regs_acquire();
                    compute_silu_tile(row_offset + n_block_start + n, 0U);
                    compute_silu_tile(row_offset + n_block_start + n + 1U, 1U);
                    tile_regs_commit();
                    pack_and_push_block(cb_m_out_idx, 2U);
                }
                if (n < n_block_size) {
                    tile_regs_acquire();
                    compute_silu_tile(row_offset + n_block_start + n, 0U);
                    tile_regs_commit();
                    pack_and_push(0U, cb_m_out_idx);
                }
                if (n_block_size < block_size) {
                    tile_regs_acquire();
                    tile_regs_commit();
                    pack_and_push_block(cb_m_out_idx, block_size - n_block_size);
                }
            }
        }

        cb_pop_front(cb_xw1_acc_idx, acc_tiles_total);
        cb_pop_front(cb_xw3_acc_idx, acc_tiles_total);
    }
}
