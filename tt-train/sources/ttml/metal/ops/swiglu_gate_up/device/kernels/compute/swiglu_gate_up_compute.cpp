// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

// ----------------------------------------------------------------------
// SwiGLU Gate-Up Compute Kernel (XW1, XW3, then M = SiLU(XW1) * XW3)
//
// Phase A: XW1[r,:], XW3[r,:] accumulated across K-blocks using L1 acc
//          directly into cb_xw1_acc / cb_xw3_acc. One N-block of W1/W3
//          at a time; block_h M-rows per sync (M sub-blocking only).
// Phase B: M[r, :] = SiLU(XW1[r, :]) * XW3[r, :] for block_h rows, pack to cb_m_out.
// ----------------------------------------------------------------------
//
// ========================= Compute kernel structure =========================
// for m_block in m_blocks:
//   reserve cb_xw1_acc, cb_xw3_acc  (block_h * per_core_N_rounded tiles each)
//   for k_block in k_blocks:
//     load X[m_block, k_block]  # block_h rows × block_size K tiles
//     for n_block in n_blocks:
//       load W1[k_block, n_block]
//       for r in block_h:
//         XW1[r, n_block] += X[r, k_block] @ W1[k_block, n_block]   # L1 acc
//       load W3[k_block, n_block]
//       for r in block_h:
//         XW3[r, n_block] += X[r, k_block] @ W3[k_block, n_block]   # L1 acc
//   push cb_xw1_acc, cb_xw3_acc
//   for r in block_h:
//     for n in per_core_N:
//       M[r, n] = SiLU(XW1[r, n]) * XW3[r, n]
//   pop cb_xw1_acc, cb_xw3_acc
// ============================================================================

constexpr uint32_t per_core_N = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);
constexpr uint32_t block_h = get_compile_time_arg_val(3);
constexpr uint32_t num_m_blocks = get_compile_time_arg_val(4);

// Derived from per_core_N and block_size (single source of truth).
constexpr uint32_t per_core_N_rounded = ((per_core_N + block_size - 1U) / block_size) * block_size;
constexpr uint32_t num_n_blocks = per_core_N_rounded / block_size;

constexpr uint32_t tiles_per_n_block = block_size * block_size;
constexpr uint32_t x_tiles_per_block = block_h * block_size;
constexpr uint32_t acc_tiles_total = block_h * per_core_N_rounded;
constexpr uint32_t in1_k_stride = block_size;

constexpr auto cb_in0_idx = tt::CBIndex::c_0;
constexpr auto cb_w1_idx = tt::CBIndex::c_1;
constexpr auto cb_w3_idx = tt::CBIndex::c_2;
constexpr auto cb_xw1_acc_idx = tt::CBIndex::c_3;
constexpr auto cb_xw3_acc_idx = tt::CBIndex::c_4;
constexpr auto cb_m_out_idx = tt::CBIndex::c_5;
constexpr auto cb_sigmoid_idx = tt::CBIndex::c_6;
constexpr auto cb_silu_idx = tt::CBIndex::c_7;

// Matmul one row: multiply X row by one N-block of weights in CB.
// in1_stride: distance between consecutive K-rows in CB (= block_size).
inline void matmul_one_row(
    const tt::CBIndex cb_x_idx,
    const tt::CBIndex cb_w_idx,
    const tt::CBIndex cb_acc_idx,
    const uint32_t x_row_offset,
    const uint32_t acc_offset,
    const uint32_t in1_start,
    const uint32_t in1_stride,
    const uint32_t k_block_size) {
    tile_regs_acquire();

    uint32_t in0_index = x_row_offset;
    uint32_t in1_index = in1_start;
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
        in1_index += in1_stride;
    }

    tile_regs_commit();
    pack_block_to_cb(static_cast<uint32_t>(cb_acc_idx), block_size, acc_offset);
}

inline void compute_sigmoid(uint32_t tile_offset, uint32_t current_block_size) {
    tile_regs_acquire();
    for (uint32_t block_idx = 0U; block_idx < current_block_size; ++block_idx) {
        copy_tile_init(cb_xw1_acc_idx);
        copy_tile(cb_xw1_acc_idx, tile_offset + block_idx, block_idx);
        sigmoid_tile_init();
        sigmoid_tile(block_idx);
    }
    tile_regs_commit();
    pack_and_push_block(cb_sigmoid_idx, block_size);
}

inline void compute_silu(uint32_t tile_offset, uint32_t current_block_size) {
    cb_wait_front(cb_sigmoid_idx, block_size);
    tile_regs_acquire();
    for (uint32_t block_idx = 0U; block_idx < current_block_size; ++block_idx) {
        mul_tiles_init(cb_xw1_acc_idx, cb_sigmoid_idx);
        mul_tiles(cb_xw1_acc_idx, cb_sigmoid_idx, tile_offset + block_idx, block_idx, block_idx);
    }
    tile_regs_commit();
    pack_and_push_block(cb_silu_idx, block_size);
    cb_pop_front(cb_sigmoid_idx, block_size);
}

inline void compute_m(uint32_t tile_offset, uint32_t current_block_size) {
    cb_wait_front(cb_silu_idx, block_size);
    tile_regs_acquire();
    for (uint32_t block_idx = 0U; block_idx < current_block_size; ++block_idx) {
        mul_tiles_init(cb_silu_idx, cb_xw3_acc_idx);
        mul_tiles(cb_silu_idx, cb_xw3_acc_idx, block_idx, tile_offset + block_idx, block_idx);
    }
    tile_regs_commit();
    pack_and_push_block(cb_m_out_idx, block_size);
    cb_pop_front(cb_silu_idx, block_size);
}

// Process one N-block of weights for one weight matrix (W1 or W3).
inline void matmul_rows_for_one_n_block(
    const tt::CBIndex cb_w_idx,
    const tt::CBIndex cb_acc_idx,
    const uint32_t n_block_offset,
    const uint32_t k_block_size,
    const bool first_k_block) {
    mm_block_init_short(
        cb_in0_idx, cb_w_idx, /*transpose=*/false, /*ct_dim=*/block_size, /*rt_dim=*/1U, /*kt_dim=*/k_block_size);
    pack_reconfig_data_format(cb_acc_idx);
    pack_reconfig_l1_acc(first_k_block ? 0 : 1U);

    for (uint32_t m_sub = 0U; m_sub < block_h; ++m_sub) {
        matmul_one_row(
            cb_in0_idx,
            cb_w_idx,
            cb_acc_idx,
            m_sub * block_size,
            m_sub * per_core_N_rounded + n_block_offset,
            /*in1_start=*/0U,
            in1_k_stride,
            k_block_size);
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

            for (uint32_t n_block = 0U; n_block < num_n_blocks; ++n_block) {
                const uint32_t n_block_offset = n_block * block_size;

                cb_wait_front(cb_w1_idx, tiles_per_n_block);
                matmul_rows_for_one_n_block(cb_w1_idx, cb_xw1_acc_idx, n_block_offset, k_block_size, first_k_block);
                cb_pop_front(cb_w1_idx, tiles_per_n_block);

                cb_wait_front(cb_w3_idx, tiles_per_n_block);
                matmul_rows_for_one_n_block(cb_w3_idx, cb_xw3_acc_idx, n_block_offset, k_block_size, first_k_block);
                cb_pop_front(cb_w3_idx, tiles_per_n_block);
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

            for (uint32_t n_block = 0U; n_block < num_n_blocks; ++n_block) {
                const uint32_t n_block_offset = n_block * block_size;
                const uint32_t current_block_size = std::min(block_size, per_core_N - n_block_offset);
                const uint32_t tile_offset = row_offset + n_block_offset;

                compute_sigmoid(tile_offset, current_block_size);
                compute_silu(tile_offset, current_block_size);
                compute_m(tile_offset, current_block_size);
            }
        }

        cb_pop_front(cb_xw1_acc_idx, acc_tiles_total);
        cb_pop_front(cb_xw3_acc_idx, acc_tiles_total);
    }
}
