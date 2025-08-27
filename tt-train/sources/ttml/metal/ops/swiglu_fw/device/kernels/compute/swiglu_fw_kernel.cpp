// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#include <compute_kernel_api/eltwise_binary_sfpu.h>

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/ops/common/compute_utils.hpp"
namespace NAMESPACE {

// ----------------------------------------------------------------------
// Problem:
//
// Given:
//   X   shape [R, P]
//   W1  shape [P, K]
//   W2  shape [K, C]
// We want:
//   Y[r, c] = sum_k( M[r, k] * W2[k, c] )
//   where
//       M[r, k] = sum_p( X[r, p] * W1[p, k] )
//
// We compute in 3 nested block loops over c, k, p
//   p is split into p_blocks for streaming X and W1
//   k is split into k_blocks for streaming M and W2
//   c is split into c_blocks to fit Y accumulators in registers
// ----------------------------------------------------------------------

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);         // total C
constexpr uint32_t hidden_Wt = get_compile_time_arg_val(3);  // total K

// Circular buffer indices
constexpr auto cb_input_idx = tt::CBIndex::c_0;      // X[r, p_block]
constexpr auto cb_w1_idx = tt::CBIndex::c_1;         // W1[p_block, k]
constexpr auto cb_w2_idx = tt::CBIndex::c_2;         // W2[k_block, c_block]
constexpr auto cb_xw1_idx = tt::CBIndex::c_6;        // M[r, k_block]
constexpr auto cb_y_idx = tt::CBIndex::c_7;          // Final Y[r, c_block]
constexpr auto cb_y_partial_idx = tt::CBIndex::c_8;  // Partial Y[r, c_block] between k_blocks

// ============================================================================
// Phase A: Compute full M[r, k_local] over all p_blocks
//   M[r, k_local] = sum_p( X[r, p] * W1[p, k_local] )
// ============================================================================
inline void compute_full_M_for_k(uint32_t k_local) {
    DPRINT << "------Computing full M for k_local=" << k_local << ENDL();
    const uint32_t sum_reg = 3;  // REG3 will hold the accumulated M[r, k_local]
    tile_regs_acquire();         // acquire working regs 0..block_size-1 and sum_reg

    fill_tile(sum_reg, 0.0f);  // initialise accumulator with 0

    // M[r, k_local] accumulator is REG3
    // For each p_block:
    //   REGt = matmul( X[r, p_local], W1[p_local, k_local] )
    //   REG3 += REGt   for t in [0, p_block_size)
    for (uint32_t p_block_start = 0; p_block_start < hidden_Wt; p_block_start += block_size) {
        const uint32_t p_block_size =
            (p_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - p_block_start;
        // DPRINT << "        p_block_start=" << p_block_start << " p_block_size=" << p_block_size << ENDL();
        // DPRINT << "        Reading X and W1 for p_block starting at " << p_block_start << ENDL();
        cb_wait_front(cb_input_idx, p_block_size);
        cb_wait_front(cb_w1_idx, p_block_size);
        // DPRINT << "        Finished reading X and W1 for p_block starting at " << p_block_start << ENDL();

        // Compute partial product tiles for this p_block into regs 0..p_block_size-1
        // DPRINT << "        Computing partial products for p_block starting at " << p_block_start << ENDL();
        mm_init(cb_input_idx, cb_w1_idx, cb_xw1_idx);
        for (uint32_t block_idx = 0; block_idx < p_block_size; ++block_idx) {
            matmul_tiles(cb_input_idx, cb_w1_idx, block_idx, block_idx, block_idx, false);
        }

        // Accumulate each partial tile into sum_reg
        add_binary_tile_init();
        for (uint32_t block_idx = 0; block_idx < p_block_size; ++block_idx) {
            add_binary_tile(sum_reg, block_idx);  // sum_reg += REG(block_idx)
        }
        // DPRINT << "        Finished computing partial products for p_block starting at " << p_block_start << ENDL();

        // DPRINT << "        Popping X and W1 for p_block starting at " << p_block_start << ENDL();
        cb_pop_front(cb_input_idx, p_block_size);
        cb_pop_front(cb_w1_idx, p_block_size);
        // DPRINT << "        Finished popping X and W1 for p_block starting at " << p_block_start << ENDL();
    }
    tile_regs_commit();

    // At this point, sum_reg (REG3) holds full M[r, k_local]
    pack_and_push(cb_xw1_idx, sum_reg);
    DPRINT << "------Finished computing full M for k_local=" << k_local << ENDL();
}

// ============================================================================
// Phase B: Accumulate into Y[r, c:c+c_block_size]
//   Y[r, c_local] += sum_k( M[r,k] * W2[k, c_local] )
// ============================================================================
// Phase B: Y[r, c_local] += sum_k( M[r,k] * W2[k, c_local] )
inline void mul_MxW2_accumulate_Y(uint32_t k_block_size, uint32_t c_block_size, bool first_k_block) {
    const uint32_t sum_reg = 3;  // working register for temp matmul result

    cb_wait_front(cb_xw1_idx, k_block_size);
    cb_wait_front(cb_w2_idx, c_block_size);

    for (uint32_t k_local = 0; k_local < k_block_size; ++k_local) {
        for (uint32_t c_local = 0; c_local < c_block_size; ++c_local) {
            mm_init(cb_xw1_idx, cb_w2_idx, cb_y_idx);
            matmul_tiles(cb_xw1_idx, cb_w2_idx, k_local, c_local, sum_reg, false);

            add_binary_tile_init();
            add_binary_tile(c_local, sum_reg);  // REGc_local += sum_reg
        }
    }

    cb_pop_front(cb_xw1_idx, k_block_size);
    cb_pop_front(cb_w2_idx, c_block_size);
}

// ============================================================================
// Spill/load partial Y[r, c_block] between k_blocks
// ============================================================================
inline void store_partial_Y_block_to_cb_y_partial(uint32_t c_block_size) {
    pack_and_push_block(cb_y_partial_idx, c_block_size);
}
inline void store_final_Y_block_to_cb_y(uint32_t c_block_size) {
    pack_and_push_block(cb_y_idx, c_block_size);
}
inline void load_partial_Y_block_from_cb_y_partial(uint32_t c_block_size) {
    cb_wait_front(cb_y_partial_idx, c_block_size);
    for (uint32_t c_local = 0; c_local < c_block_size; ++c_local) {
        copy_tile(cb_y_partial_idx, /* tile_idx */ c_local, /* dst */ c_local);  // CB tile -> REG
    }
    cb_pop_front(cb_y_partial_idx, c_block_size);
}

// ============================================================================
// Main kernel: performs
// for r in rows:
//   for c in c_blocks:
//     init Y_block (regs)
//     for k in k_blocks:
//       compute M_block = { M[r, k_local] = sum_p(...) }
//       Y_block += sum_k( M_block * W2_block )
//     store final Y_block to cb_y
// ============================================================================
inline void MAIN {
    // return;
    init_sfpu(cb_input_idx, cb_y_idx);
    binary_op_init_common(cb_input_idx, cb_w1_idx, cb_y_idx);

    for (uint32_t r = 0; r < num_rows_per_core; ++r) {
        // Loop over c_blocks with tail handling
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            // const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : Wt - c_block_start;

            // Loop over k_blocks
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;

                const bool first_k_block = (k_block_start == 0);
                const bool last_k_block = (k_block_start + block_size >= hidden_Wt);

                // ---- Phase A: compute all M[r,k] for this k_block ----
                // M[r,k] = sum_p( X[r,p] * W1[p,k] )
                for (uint32_t k_local = 0; k_local < k_block_size; ++k_local) {
                    compute_full_M_for_k(k_local);
                }
                continue;
                /*
                // ---- Phase B: accumulate into Y[r, c_block] ----
                tile_regs_acquire();  // regs for Y accumulators
                if (first_k_block) {
                    // Clear accumulators if first_k_block
                    for (uint32_t c_local = 0; c_local < c_block_size; ++c_local) {
                        fill_tile(c_local, 0.0f);  // REGc_local = 0
                    }
                } else {
                    load_partial_Y_block_from_cb_y_partial(c_block_size);
                }
                mul_MxW2_accumulate_Y(k_block_size, c_block_size, first_k_block);

                // ---- Spill or store final ----
                tile_regs_commit();
                if (last_k_block) {
                    store_final_Y_block_to_cb_y(c_block_size);
                } else {
                    store_partial_Y_block_to_cb_y_partial(c_block_size);
                }
                */
            }
        }
    }
}

}  // namespace NAMESPACE
