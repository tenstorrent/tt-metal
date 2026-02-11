// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <compute_kernel_api/eltwise_binary_sfpu.h>
#include <compute_kernel_api/reconfig_data_format.h>

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/copy_dest_values.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

// ----------------------------------------------------------------------
// SwiGLU Forward Compute Kernel with Packer L1 Accumulation
//
// Phase A: XW1/XW3 accumulate across p_blocks using L1 acc directly into
//          cb_xw1/cb_xw3 (no partial CBs needed)
// Phase B: SiLU activation (unchanged)
// Phase C: Y accumulate across k_blocks using L1 acc directly into cb_y
//          (no cb_y_partial needed)
// ----------------------------------------------------------------------

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t max_rows_for_sync = get_compile_time_arg_val(1);
constexpr uint32_t block_size = get_compile_time_arg_val(2);
constexpr uint32_t Wt = get_compile_time_arg_val(3);
constexpr uint32_t hidden_Wt = get_compile_time_arg_val(4);

const uint32_t hidden_Wt_rounded_up = ((hidden_Wt + block_size - 1) / block_size) * block_size;

// CBs with input data
constexpr auto cb_input_idx = tt::CBIndex::c_0;
constexpr auto cb_w1_idx = tt::CBIndex::c_1;
constexpr auto cb_w2_idx = tt::CBIndex::c_2;
constexpr auto cb_w3_idx = tt::CBIndex::c_3;
// CBs with intermediate computations - L1 acc eliminates partial CBs
// c_4 and c_5 are no longer used (were cb_xw1_partial, cb_xw3_partial)
constexpr auto cb_xw1_idx = tt::CBIndex::c_6;  // (X @ W1)[r, :] - L1 acc target
constexpr auto cb_xw3_idx = tt::CBIndex::c_7;  // (X @ W3)[r, :] - L1 acc target
constexpr auto cb_m_idx = tt::CBIndex::c_8;    // M[r, k_block]
// c_9 is no longer used (was cb_y_partial)
// CB with output data
constexpr auto cb_y_idx = tt::CBIndex::c_10;  // Y[r, c_block] - L1 acc target

// ============================================================================
// Phase A: Accumulate X @ W into final CB using packer L1 accumulation.
// Packs results at offset k_block_start in the final CB.
// first_p_block: l1_acc=0 (seed), subsequent: l1_acc=1 (accumulate).
// ============================================================================
inline void mul_XW_accumulate_l1(
    const tt::CBIndex cb_x_idx,
    const tt::CBIndex cb_w_idx,
    const tt::CBIndex cb_out_idx,  // Final CB (cb_xw1 or cb_xw3), already reserved
    const uint32_t k_block_start,  // Offset in final CB for this k_block
    const uint32_t p_block_size,
    const bool first_p_block) {
    tile_regs_acquire();

    // Wait for W tiles (batched: block_size rows × block_size tiles)
    constexpr uint32_t tiles_per_batch = block_size * block_size;
    cb_wait_front(cb_w_idx, tiles_per_batch);

    mm_block_init_short(
        cb_x_idx,
        cb_w_idx,
        /*transpose=*/false,
        /*ct_dim=*/block_size,
        /*rt_dim=*/1,
        /*kt_dim=*/p_block_size);

    uint32_t in0_index = 0;
    uint32_t in1_index = 0;
    for (uint32_t p = 0; p < p_block_size; ++p) {
        matmul_block(
            cb_x_idx,
            cb_w_idx,
            in0_index,
            in1_index,
            /*dst_index=*/0,
            /*transpose=*/false,
            /*ct_dim=*/block_size,
            /*rt_dim=*/1,
            /*kt_dim=*/p_block_size);
        in0_index++;
        in1_index += block_size;
    }

    cb_pop_front(cb_w_idx, tiles_per_batch);

    tile_regs_commit();
    tile_regs_wait();

    // Pack to final CB at correct k_block offset using L1 accumulation
    pack_reconfig_data_format(cb_out_idx);
    PACK((llk_pack_reconfig_l1_acc(first_p_block ? 0 : 1)));
    for (uint32_t k = 0; k < block_size; ++k) {
        pack_tile<true>(k, cb_out_idx, k_block_start + k);
    }
    // Disable L1 acc after packing (will be re-enabled on next call if needed)
    PACK((llk_pack_reconfig_l1_acc(0)));

    tile_regs_release();
}

// ============================================================================
// Compute XW1 and XW3 for one row using L1 accumulation.
// Reserves cb_xw1/cb_xw3 once, accumulates across all p_blocks, pushes once.
// ============================================================================
inline void compute_XW1_XW3_for_r() {
    // Reserve full rows in final CBs once
    cb_reserve_back(cb_xw1_idx, hidden_Wt_rounded_up);
    cb_reserve_back(cb_xw3_idx, hidden_Wt_rounded_up);

    for (uint32_t p_block_start = 0; p_block_start < Wt; p_block_start += block_size) {
        const uint32_t p_block_size = (p_block_start + block_size <= Wt) ? block_size : Wt - p_block_start;
        const bool first_p_block = (p_block_start == 0);

        cb_wait_front(cb_input_idx, block_size);

        for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
            mul_XW_accumulate_l1(cb_input_idx, cb_w1_idx, cb_xw1_idx, k_block_start, p_block_size, first_p_block);

            mul_XW_accumulate_l1(cb_input_idx, cb_w3_idx, cb_xw3_idx, k_block_start, p_block_size, first_p_block);
        }

        cb_pop_front(cb_input_idx, block_size);
    }

    // All p_blocks done, push the completed rows
    cb_push_back(cb_xw1_idx, hidden_Wt_rounded_up);
    cb_push_back(cb_xw3_idx, hidden_Wt_rounded_up);
}

// ============================================================================
// Compute M[r, :] = SiLU(XW1[r, :]) * XW3[r, :]
// ============================================================================
inline void compute_M_for_r() {
    cb_wait_front(cb_xw1_idx, hidden_Wt_rounded_up);
    cb_wait_front(cb_xw3_idx, hidden_Wt_rounded_up);

    for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
        const uint32_t k_block_size =
            (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;

        for (uint32_t k = 0; k < k_block_size; ++k) {
            constexpr uint32_t xw1_reg = 0U;
            constexpr uint32_t xw3_reg = 1U;
            constexpr uint32_t silu_reg = 2U;
            constexpr uint32_t m_reg = 3U;

            const uint32_t tile_offset = k_block_start + k;

            tile_regs_acquire();
            copy_tile_init(cb_xw1_idx);
            copy_tile(cb_xw1_idx, tile_offset, xw1_reg);
            copy_tile_init(cb_xw3_idx);
            copy_tile(cb_xw3_idx, tile_offset, xw3_reg);

            copy_dest_values_init();
            copy_dest_values(xw1_reg, silu_reg);
            sigmoid_tile_init();
            sigmoid_tile(silu_reg);
            mul_binary_tile_init();
            mul_binary_tile(xw1_reg, silu_reg, silu_reg);
            mul_binary_tile(silu_reg, xw3_reg, m_reg);

            tile_regs_commit();

            cb_reserve_back(cb_m_idx, 1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_m_idx);
            pack_tile(m_reg, cb_m_idx);
            tile_regs_release();
            cb_push_back(cb_m_idx, 1);
        }
        if (k_block_size != block_size) {
            // Push padding M tiles
            for (uint32_t pad = 0; pad < block_size - k_block_size; ++pad) {
                cb_reserve_back(cb_m_idx, 1);
                tile_regs_acquire();
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_m_idx);
                tile_regs_release();
                cb_push_back(cb_m_idx, 1);
            }
        }
    }

    cb_pop_front(cb_xw1_idx, hidden_Wt_rounded_up);
    cb_pop_front(cb_xw3_idx, hidden_Wt_rounded_up);
}

// ============================================================================
// Phase C: Accumulate Y += M @ W2 for one c_block using L1 accumulation.
// Accumulates across k_blocks directly into cb_y (no partial CB).
// ============================================================================
inline void mul_MW2_accumulate_Y_l1(
    const uint32_t k_block_start, const uint32_t k_block_size, const uint32_t c_block_size, const bool first_k_block) {
    tile_regs_acquire();

    // Wait for W2 batch
    constexpr uint32_t tiles_per_batch = block_size * block_size;
    cb_wait_front(cb_w2_idx, tiles_per_batch);

    mm_init_short(cb_m_idx, cb_w2_idx, false);

    // Compute Y[c] += sum_k M[k] * W2[k, c]
    for (uint32_t c = 0; c < c_block_size; ++c) {
        const uint32_t cb_col_offset = c * block_size;
        for (uint32_t k = 0; k < k_block_size; ++k) {
            matmul_tiles(cb_m_idx, cb_w2_idx, k_block_start + k, cb_col_offset + k, c);
        }
    }

    cb_pop_front(cb_w2_idx, tiles_per_batch);

    tile_regs_commit();
    tile_regs_wait();

    // Pack to cb_y using L1 accumulation
    pack_reconfig_data_format(cb_y_idx);
    PACK((llk_pack_reconfig_l1_acc(first_k_block ? 0 : 1)));
    for (uint32_t c = 0; c < block_size; ++c) {
        pack_tile<true>(c, cb_y_idx, c);
    }
    PACK((llk_pack_reconfig_l1_acc(0)));

    tile_regs_release();
}

// ============================================================================
// Main kernel
// ============================================================================
void kernel_main() {
    init_sfpu(cb_input_idx, cb_y_idx);
    binary_op_init_common(cb_input_idx, cb_w1_idx, cb_y_idx);

    for (uint32_t r = 0; r < max_rows_for_sync; ++r) {
        const bool is_padding_row = (r >= num_rows_per_core);

        // ---- Phase A: Accumulate XW1[r,:] and XW3[r,:] with L1 acc ----
        compute_XW1_XW3_for_r();

        // ---- Phase B: Compute M[r,:] ----
        compute_M_for_r();
        cb_wait_front(cb_m_idx, hidden_Wt_rounded_up);

        // ---- Phase C: Compute Y[r,:] = M @ W2 with L1 acc ----
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : Wt - c_block_start;

            if (is_padding_row) {
                // Consume W2 to stay in sync
                for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                    constexpr uint32_t tiles_per_batch = block_size * block_size;
                    cb_wait_front(cb_w2_idx, tiles_per_batch);
                    cb_pop_front(cb_w2_idx, tiles_per_batch);
                }
            } else {
                // Reserve cb_y once per c_block, accumulate across k_blocks
                cb_reserve_back(cb_y_idx, block_size);

                for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                    const uint32_t k_block_size =
                        (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;
                    const bool first_k_block = (k_block_start == 0);

                    mul_MW2_accumulate_Y_l1(k_block_start, k_block_size, c_block_size, first_k_block);
                }

                // All k_blocks done for this c_block, push Y
                cb_push_back(cb_y_idx, block_size);
            }
        }

        cb_pop_front(cb_m_idx, hidden_Wt_rounded_up);
    }
}
