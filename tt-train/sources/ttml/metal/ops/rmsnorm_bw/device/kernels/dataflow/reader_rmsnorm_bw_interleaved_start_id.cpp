// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// CBs with input data
constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;
constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;
constexpr uint32_t cb_rms_a_idx = tt::CBIndex::c_4;
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;
constexpr uint32_t cb_mat_mul_reduce = tt::CBIndex::c_6;
constexpr uint32_t cb_zero_idx = tt::CBIndex::c_7;
constexpr uint32_t cb_gate_idx = tt::CBIndex::c_12;
// CBs with output data
constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_8;
constexpr uint32_t cb_dL_dgamma_components = tt::CBIndex::c_9;
// CBs with intermediate computations
constexpr uint32_t cb_recip_rms_a_bcasted_idx = tt::CBIndex::c_10;
constexpr uint32_t cb_scale_idx = tt::CBIndex::c_11;

constexpr uint32_t packed_scaler = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t mask_w = get_compile_time_arg_val(2);
constexpr uint32_t Wt = get_compile_time_arg_val(3);
// The layout of dL_dout and the gate for head_merged_row; 1 and 1 when ungated.
constexpr uint32_t layout_Ht = get_compile_time_arg_val(4);
constexpr uint32_t layout_num_heads = get_compile_time_arg_val(5);

#ifdef DO_MASK_W
constexpr bool do_mask_w = true;
#else
constexpr bool do_mask_w = false;
#endif

#ifdef GATED
constexpr bool gated = true;
#else
constexpr bool gated = false;
#endif

#if defined(GATED) && !defined(EVERYTHING_FITS_IN_L1)
#error "Gated mode needs EVERYTHING_FITS_IN_L1, like the compute kernel: streaming would read the gate twice."
#endif

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t gamma_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t rms_a_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dL_out_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t gate_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
    constexpr uint16_t zero = 0x0;
    constexpr uint16_t minus_inf = 0xFF80;  // (bfloat16)-inf -> uint16_t
    // Generate mask tile.
    if constexpr (do_mask_w) {
        generate_mask_tile(cb_mask_w_idx, one, zero, mask_w);
    }
    // Generate tile with scalar.
    generate_tile_with_packed_bfloat16_value(cb_scaler_idx, packed_scaler);
    // Generate tile for matmul row reduce.
    generate_matmul_row_reduce_tile(cb_mat_mul_reduce);
    // Generate tile with zeros.
    generate_tile_with_bfloat16_value(cb_zero_idx, zero);

    // The input may be fp32; every other tensor read here is bf16.
    const uint32_t input_tile_bytes = get_tile_size(cb_input_idx);
    const uint32_t tile_bytes = get_tile_size(cb_dL_out_idx);
    constexpr auto input_args = TensorAccessorArgs<6>();
    constexpr auto gamma_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto rms_a_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr auto dL_out_args = TensorAccessorArgs<rms_a_args.next_compile_time_args_offset()>();
    constexpr auto gate_args = TensorAccessorArgs<dL_out_args.next_compile_time_args_offset()>();
    const auto input_address_generator = TensorAccessor(input_args, input_address);
    const auto gamma_address_generator = TensorAccessor(gamma_args, gamma_address);
    const auto rms_a_address_generator = TensorAccessor(rms_a_args, rms_a_address);
    const auto dL_out_address_generator = TensorAccessor(dL_out_args, dL_out_address);
    const auto gate_address_generator = TensorAccessor(gate_args, gate_address);

    // Read input tensors row by row, reading each row's data twice due to compute requirements
    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        // Read RMS(a) once per row - shape [B,1,S,1], one scalar per row
        read_tiles_by_row(cb_rms_a_idx, rms_a_address_generator, r, 1, tile_bytes, 1);

        const uint32_t merged_row_start = head_merged_row<layout_Ht, layout_num_heads>(r) * Wt;
        // Issues the reads of the input, dL_dout and gate blocks at column c; the caller barriers and pushes them.
        auto read_row_blocks = [&](const uint32_t c) {
            read_tiles_by_row</* UseBarrier = */ false>(
                cb_input_idx, input_address_generator, (r * Wt) + c, block_size, input_tile_bytes, block_size);
            read_tiles_by_row</* UseBarrier = */ false>(
                cb_dL_out_idx, dL_out_address_generator, merged_row_start + c, block_size, tile_bytes, block_size);
            if constexpr (gated) {
                read_tiles_by_row</* UseBarrier = */ false>(
                    cb_gate_idx, gate_address_generator, merged_row_start + c, block_size, tile_bytes, block_size);
            }
        };
        auto push_row_blocks = []() {
            cb_push_back(cb_input_idx, block_size);
            cb_push_back(cb_dL_out_idx, block_size);
            if constexpr (gated) {
                cb_push_back(cb_gate_idx, block_size);
            }
        };

        // First pass: read row data for first compute phase
        for (uint32_t c = 0; c < Wt; c += block_size) {
            read_row_blocks(c);
#ifndef EVERYTHING_FITS_IN_L1
            // If not everything fits in L1, we need to reread gamma for each row.
            read_tiles_by_row(cb_gamma_idx, gamma_address_generator, c, block_size, tile_bytes, block_size);
            // Barrier called by read_tiles_by_row with UseBarrier=true above
            push_row_blocks();
#else
            // If everything fits in L1, we can read gamma only once.
            if (r == start_row) {
                read_tiles_by_row(cb_gamma_idx, gamma_address_generator, c, block_size, tile_bytes, block_size);
                // Barrier called by read_tiles_by_row with UseBarrier=true above
            } else {
                noc_async_read_barrier();
            }
            push_row_blocks();
#endif
        }

#ifndef EVERYTHING_FITS_IN_L1
        // Second pass: read row data again for second compute phase
        for (uint32_t c = 0; c < Wt; c += block_size) {
            read_row_blocks(c);
            read_tiles_by_row(cb_gamma_idx, gamma_address_generator, c, block_size, tile_bytes, block_size);
            // Barrier called by read_tiles_by_row with UseBarrier=true above
            push_row_blocks();
        }
#endif
    }
}
