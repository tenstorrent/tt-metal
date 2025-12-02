// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// CBs with input data
constexpr auto cb_input_idx = tt::CBIndex::c_0;  // X[r, p_block]
constexpr auto cb_w1_idx = tt::CBIndex::c_1;     // W1[p_block, k]
constexpr auto cb_w2_idx = tt::CBIndex::c_2;     // W2[k_block, c_block]
constexpr auto cb_w3_idx = tt::CBIndex::c_3;     // W3[p_block, k]
// CBs with intermediate computations
constexpr auto cb_xw1_partial_idx = tt::CBIndex::c_4;  // Partial (X @ W1)[r, k_block] between p_blocks
constexpr auto cb_xw3_partial_idx = tt::CBIndex::c_5;  // Partial (X @ W3)[r, k_block] between p_blocks
constexpr auto cb_xw1_idx = tt::CBIndex::c_6;          // (X @ W1)[r, k_block]
constexpr auto cb_xw3_idx = tt::CBIndex::c_7;          // (X @ W3)[r, k_block]
constexpr auto cb_m_idx = tt::CBIndex::c_8;            // M[r, k_block]
constexpr auto cb_y_partial_idx = tt::CBIndex::c_9;    // Partial Y[r, c_block] between k_blocks
// CB with output data
constexpr auto cb_y_idx = tt::CBIndex::c_10;  // Final Y[r, c_block]

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);         // output width (C)
constexpr uint32_t hidden_Wt = get_compile_time_arg_val(2);  // inner dimension (K==P)

void kernel_main() {
    uint32_t ra = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(ra++);  // DRAM base for X
    uint32_t w1_address = get_arg_val<uint32_t>(ra++);     // DRAM base for W1
    uint32_t w2_address = get_arg_val<uint32_t>(ra++);     // DRAM base for W2
    uint32_t w3_address = get_arg_val<uint32_t>(ra++);     // DRAM base for W3
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(ra++);
    uint32_t start_row = get_arg_val<uint32_t>(ra++);

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);

    // Address generators
    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto w1_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w2_args = TensorAccessorArgs<w1_args.next_compile_time_args_offset()>();
    constexpr auto w3_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();
    const auto x_address_generator = TensorAccessor(x_args, input_address, tile_bytes);
    const auto w1_address_generator = TensorAccessor(w1_args, w1_address, tile_bytes);
    const auto w2_address_generator = TensorAccessor(w2_args, w2_address, tile_bytes);
    const auto w3_address_generator = TensorAccessor(w3_args, w3_address, tile_bytes);

    const uint32_t end_row = start_row + num_rows_to_process;

#ifdef ROW_OF_M_FITS_IN_L1
    // ================== Loop structure matches compute kernel ==================
    // Flash-attention optimization: for r in rows:
    //     # Phase A: Compute XW1[r, :] and XW3[r, :] - read X[r, p_block] only once!
    //     for p_block in p_blocks:                    # OUTER LOOP - read X once per p_block
    //        read X[r, p_block]
    //        for k_block in k_blocks:                 # INNER LOOP - accumulate across full hidden dimension
    //          # First process W1 for all p in p_block (compute processes W1 first)
    //          for p in p_block:
    //            read W1[p, k_block] (read entire row of k_block elements)
    //          # Then process W3 for all p in p_block (compute processes W3 second)
    //          for p in p_block:
    //            read W3[p, k_block] (read entire row of k_block elements)
    //          # Compute processes: XW1[k] += X[r,p] * W1[p,k] for all p, then XW3[k] += X[r,p] * W3[p,k] for all p
    //
    //     # Phase B: Compute M[r,:] once
    //     [no reading required to compute M[r, :]]
    //
    //     # Phase C: Use M[r, :] for all c-blocks to compute Y[r, :]
    //     for c_block in c_blocks:
    //        for k_block in k_blocks:
    //           read W2[k_block, c_block]
    //           [compute Y[r, c_block] += M[r, k_block] * W2[k_block, c_block]]
    // ============================================================================
    for (uint32_t r = start_row; r < end_row; ++r) {
        // ---- Phase A: Compute XW1[r,:] and XW3[r,:] with flash-attention optimization ----
        // XW1[r,k] = sum_p( X[r,p] * W1[p,k] )
        // XW3[r,k] = sum_p( X[r,p] * W3[p,k] )
        for (uint32_t p_block_start = 0; p_block_start < Wt; p_block_start += block_size) {
            const uint32_t p_block_size = (p_block_start + block_size <= Wt) ? block_size : Wt - p_block_start;

            // --- Read p_block_size tiles of X[r, p_block] ONCE per p_block
            uint32_t x_tile_start = r * Wt + p_block_start;
            read_tiles_by_row(cb_input_idx, x_address_generator, x_tile_start, p_block_size, tile_bytes, block_size);

            // Stream W1 and W3 data organized by k_blocks to match compute kernel expectations
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;

                // First, read all W1 data for this k_block (compute processes W1 first)
                for (uint32_t p = 0; p < p_block_size; ++p) {
                    const uint32_t p_global = p_block_start + p;
                    uint32_t w1_tile_start = p_global * hidden_Wt + k_block_start;
                    read_tiles_by_row(
                        cb_w1_idx, w1_address_generator, w1_tile_start, k_block_size, tile_bytes, block_size);
                }

                // Then, read all W3 data for this k_block (compute processes W3 second)
                for (uint32_t p = 0; p < p_block_size; ++p) {
                    const uint32_t p_global = p_block_start + p;
                    uint32_t w3_tile_start = p_global * hidden_Wt + k_block_start;
                    read_tiles_by_row(
                        cb_w3_idx, w3_address_generator, w3_tile_start, k_block_size, tile_bytes, block_size);
                }
            }
        }

        // ---- Phase B: Compute M[r, :] once ----
        // Compute M[r, k] = SiLU( XW1[r, k] ) * XW3[r, k]
        // No reading required to compute M[r, :]

        // ---- Phase C: Use M[r, :] for all c_blocks ----
        // Y[r, :] = sum_k( M[r, k] * W2[k, c] )
        for (uint32_t c_block_start = 0; c_block_start < Wt; c_block_start += block_size) {
            const uint32_t c_block_size = (c_block_start + block_size <= Wt) ? block_size : Wt - c_block_start;
            // Compute Y[r, c_block_start : c_block_start + c_block_size]
            for (uint32_t k_block_start = 0; k_block_start < hidden_Wt; k_block_start += block_size) {
                const uint32_t k_block_size =
                    (k_block_start + block_size <= hidden_Wt) ? block_size : hidden_Wt - k_block_start;
                for (uint32_t c = 0; c < c_block_size; ++c) {
                    const uint32_t c_global = c_block_start + c;
                    // Read W2[k_block, c_global] - one column of block_size elements
                    uint32_t w2_tile_start = k_block_start * Wt + c_global;
                    read_tiles_by_col(
                        cb_w2_idx, w2_address_generator, w2_tile_start, k_block_size, tile_bytes, Wt, block_size);

                    // --- Compute Y_partial[r, c_global] += M[r, k_block] * W2[k_block, c_global]
                }
            }
            // Store Y_partial[r, c_block] → Y[r, c_block]
        }
    }

#else

    // ================== Loop structure matches compute kernel ==================
    // for r in rows:
    //   for c_block in c_blocks:
    //     for k_block in k_blocks:
    //       for k in k_block:
    //         for p_block in p_blocks:
    //           stream X[r, p_block] and W1[p_block, k] for all p in p_block
    //           [compute M[r, k] = sum_p( X[r, p] * W1[p, k] )]
    //       for c in c_block:
    //         stream W2[k_block, c] for all k in k_block
    //         [compute Y[r, c] += sum_k( M[r, k] * W2[k, c] )]
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
                        // Calculate starting tile index for X. We read X in row-major order, so the offset equals
                        uint32_t x_tile_start = r * Wt + p_block_start;
                        read_tiles_by_row(
                            cb_input_idx, x_address_generator, x_tile_start, p_block_size, tile_bytes, block_size);

                        // --- Read p_block_size tiles of W1[p_block, k_global]
                        // Calculate starting tile index for W1. We read W1 in col-major order, so offset equals
                        uint32_t w1_tile_start = p_block_start * hidden_Wt + k_global;
                        read_tiles_by_col(
                            cb_w1_idx,
                            w1_address_generator,
                            w1_tile_start,
                            p_block_size,
                            tile_bytes,
                            hidden_Wt,
                            block_size);

                        // --- Read p_block_size tiles of W3[p_block, k_global]
                        // Calculate starting tile index for W3. We read W3 in col-major order, so offset equals
                        uint32_t w3_tile_start = p_block_start * hidden_Wt + k_global;
                        read_tiles_by_col(
                            cb_w3_idx,
                            w3_address_generator,
                            w3_tile_start,
                            p_block_size,
                            tile_bytes,
                            hidden_Wt,
                            block_size);
                    }
                }

                // --------------------
                // STEP B: W2 streaming will be handled by the compute kernel synchronization
                // --------------------
                // Stream W2 columns one by one as compute kernel requests them
                for (uint32_t c_local = 0; c_local < c_block_size; ++c_local) {
                    const uint32_t c_global = c_block_start + c_local;

                    // --- Read W2[k_block, c_global] - one column of block_size elements
                    // Calculate starting tile index for W2. We read W2 in col-major order, so offset equals
                    uint32_t w2_tile_start = k_block_start * Wt + c_global;
                    read_tiles_by_col(
                        cb_w2_idx, w2_address_generator, w2_tile_start, k_block_size, tile_bytes, Wt, block_size);
                }
            }
        }
    }

#endif  // ROW_OF_M_FITS_IN_L1
}
