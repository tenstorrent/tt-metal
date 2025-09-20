// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#include <debug/dprint.h>

#include <ostream>

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

// CBs with input data
constexpr auto cb_input_idx = tt::CBIndex::c_0;  // X[r, p_block]
constexpr auto cb_w1_idx = tt::CBIndex::c_1;     // W1[p_block, k]
constexpr auto cb_w2_idx = tt::CBIndex::c_2;     // W2[k_block, c_block]
constexpr auto cb_w3_idx = tt::CBIndex::c_3;     // W3[p_block, k]
// CBs with masks
constexpr auto cb_mask_w_idx = tt::CBIndex::c_4;   // Mask for input inner dimension
constexpr auto cb_mask_hw_idx = tt::CBIndex::c_5;  // Mask for hidden inner dimension
// CBs with intermediate computations
constexpr auto cb_xw1_idx = tt::CBIndex::c_6;        // (X @ W1)[r, k_block]
constexpr auto cb_xw3_idx = tt::CBIndex::c_7;        // (X @ W3)[r, k_block]
constexpr auto cb_m_idx = tt::CBIndex::c_8;          // M[r, k_block]
constexpr auto cb_y_partial_idx = tt::CBIndex::c_9;  // Partial Y[r, c_block] between k_blocks
// CB with output data
constexpr auto cb_y_idx = tt::CBIndex::c_10;  // Final Y[r, c_block]

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);         // output width (C)
constexpr uint32_t hidden_Wt = get_compile_time_arg_val(2);  // inner dimension (K==P)
constexpr uint32_t mask_w = get_compile_time_arg_val(3);     // mask for input inner dimension
constexpr uint32_t mask_hw = get_compile_time_arg_val(4);    // mask for hidden inner dimension

#ifdef DO_MASK_W
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

#ifdef DO_MASK_HW
constexpr bool do_mask_hw = true;
#else
constexpr bool do_mask_hw = false;
#endif

// TODO(maciek): Decide on one consistent version of whitespace in X[a, b] vs. X[a,b]

// TODO(maciek): Update read_tile to allign Stas's change.
// TODO(maciek): Move all read_tiles_by_row and read_tiles_by_col to common utils file, and reuse them in other
// operations. Utility: read contiguous tiles in row-major from DRAM to CB. Before calling this function, make sure to
// reserve space in the CB and after the call, push the tiles.
inline void read_tiles_by_row(
    uint32_t cb_idx,
    const InterleavedAddrGenFast<true>& addr_gen,
    uint32_t start_idx,
    uint32_t num_tiles,
    const uint32_t tile_bytes) {
    uint32_t l1_addr = get_write_ptr(cb_idx);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        noc_async_read_tile(start_idx + t, addr_gen, l1_addr);
        l1_addr += tile_bytes;
    }
}

// Utility: read contiguous tiles in col-major from DRAM to CB. Before calling this function, make sure to reserve space
// in the CB and after the call, push the tiles.
inline void read_tiles_by_col(
    uint32_t cb_idx,
    const InterleavedAddrGenFast<true>& addr_gen,
    uint32_t start_idx,
    uint32_t num_tiles,
    const uint32_t tile_bytes,
    uint32_t stride) {
    uint32_t l1_addr = get_write_ptr(cb_idx);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        uint32_t tile_idx = start_idx + t * stride;
        noc_async_read_tile(tile_idx, addr_gen, l1_addr);
        l1_addr += tile_bytes;
    }
}

void kernel_main() {
    uint32_t ra = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(ra++);  // DRAM base for X
    uint32_t w1_address = get_arg_val<uint32_t>(ra++);     // DRAM base for W1
    uint32_t w2_address = get_arg_val<uint32_t>(ra++);     // DRAM base for W2
    uint32_t w3_address = get_arg_val<uint32_t>(ra++);     // DRAM base for W3
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);
    uint32_t start_row = get_arg_val<uint32_t>(ra++);

    constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
    constexpr uint16_t zero = 0x0;
    // Generate s if needed
    if constexpr (do_mask_w) {
        generate_mask_tile(cb_mask_w_idx, one, zero, mask_w);
    }
    if constexpr (do_mask_hw) {
        generate_mask_tile(cb_mask_hw_idx, one, zero, mask_hw);
    }

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);
    const DataFormat data_fmt = get_dataformat(cb_input_idx);

    // Address generators
    const InterleavedAddrGenFast<true> x_ag = {
        .bank_base_address = input_address, .page_size = tile_bytes, .data_format = data_fmt};
    const InterleavedAddrGenFast<true> w1_ag = {
        .bank_base_address = w1_address, .page_size = tile_bytes, .data_format = data_fmt};
    const InterleavedAddrGenFast<true> w2_ag = {
        .bank_base_address = w2_address, .page_size = tile_bytes, .data_format = data_fmt};
    const InterleavedAddrGenFast<true> w3_ag = {
        .bank_base_address = w3_address, .page_size = tile_bytes, .data_format = data_fmt};

    const uint32_t end_row = start_row + num_rows_to_process;

    // ================== Loop structure matches compute kernel ==================
    // for r in rows:
    //   for c_block in c_blocks:
    //     for k_block in k_blocks:
    //       for k in k_block:
    //         for p_block in p_blocks:
    //           stream X[r, p_block] and W1[p_block, k] for all p in p_block
    //           [compute M[r,k] = sum_p( X[r,p] * W1[p,k] )]
    //       for c in c_block:
    //         stream W2[k_block, c] for all k in k_block
    //         [compute Y[r,c] += sum_k( M[r,k] * W2[k,c] )]
    // ============================================================================
    for (uint32_t r = start_row; r < end_row; ++r) {
        // Loop over c_blocks
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : (Wt - c_block_start);

            // Loop over k_blocks
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : (hidden_Wt - k_block_start);

                // --------------------
                // STEP A: Stream X + W1 for all k in this k_block
                // --------------------
                for (uint32_t k = 0; k < k_block_size; ++k) {
                    const uint32_t k_global = k_block_start + k;

                    // Loop over p_blocks along inner dimention
                    for (uint32_t p_block_start = 0; p_block_start < Wt; p_block_start += block_size) {
                        const uint32_t p_block_size =
                            (p_block_start + block_size <= Wt) ? block_size : (Wt - p_block_start);

                        // --- Read p_block_size tiles of X[r, p_block]
                        // TODO: Consider moving reserve_back and push_back to read_tiles_by_row/read_tiles_by_col
                        cb_reserve_back(cb_input_idx, block_size);
                        // Calculate starting tile index for X. We read X in row-major order, so the offset equals
                        uint32_t x_tile_start = r * Wt + p_block_start;
                        read_tiles_by_row(cb_input_idx, x_ag, x_tile_start, p_block_size, tile_bytes);
                        noc_async_read_barrier();
                        cb_push_back(cb_input_idx, block_size);

                        // --- Read p_block_size tiles of W1[p_block, k_global]
                        cb_reserve_back(cb_w1_idx, block_size);
                        // Calculate starting tile index for W1. We read W1 in col-major order, so offset equals
                        uint32_t w1_tile_start = p_block_start * hidden_Wt + k_global;
                        read_tiles_by_col(cb_w1_idx, w1_ag, w1_tile_start, p_block_size, tile_bytes, hidden_Wt);
                        noc_async_read_barrier();
                        cb_push_back(cb_w1_idx, block_size);

                        // --- Read p_block_size tiles of W3[p_block, k_global]
                        cb_reserve_back(cb_w3_idx, block_size);
                        // Calculate starting tile index for W3. We read W3 in col-major order, so offset equals
                        uint32_t w3_tile_start = p_block_start * hidden_Wt + k_global;
                        read_tiles_by_col(cb_w3_idx, w3_ag, w3_tile_start, p_block_size, tile_bytes, hidden_Wt);
                        noc_async_read_barrier();
                        cb_push_back(cb_w3_idx, block_size);
                    }
                }

                // --------------------
                // STEP B: W2 streaming will be handled by the compute kernel synchronization
                // --------------------
                // Stream W2 columns one by one as compute kernel requests them
                for (uint32_t c_local = 0; c_local < c_block_size; ++c_local) {
                    const uint32_t c_global = c_block_start + c_local;

                    // Read W2[k_block, c_global] - one column of block_size elements
                    cb_reserve_back(cb_w2_idx, block_size);
                    // Calculate starting tile index for W2. We read W2 in col-major order, so offset equals
                    uint32_t w2_tile_start = k_block_start * Wt + c_global;
                    read_tiles_by_col(cb_w2_idx, w2_ag, w2_tile_start, k_block_size, tile_bytes, Wt);
                    noc_async_read_barrier();
                    cb_push_back(cb_w2_idx, block_size);
                }
            }
        }
    }
}
