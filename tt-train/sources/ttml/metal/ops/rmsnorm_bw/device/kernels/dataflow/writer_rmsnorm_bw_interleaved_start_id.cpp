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
// CBs with output data
constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_8;
constexpr uint32_t cb_dL_dgamma_components = tt::CBIndex::c_9;
constexpr uint32_t cb_dL_dgate_idx = tt::CBIndex::c_13;
// CBs with intermediate computations
constexpr uint32_t cb_recip_rms_a_bcasted_idx = tt::CBIndex::c_10;
constexpr uint32_t cb_scale_idx = tt::CBIndex::c_11;
constexpr uint32_t cb_slice_partial_idx = tt::CBIndex::c_15;
constexpr uint32_t cb_gathered_partials_idx = tt::CBIndex::c_16;

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
// The layout of dL_dgate for head_merged_row; 1 and 1 when ungated.
constexpr uint32_t layout_Ht = get_compile_time_arg_val(2);
constexpr uint32_t layout_num_heads = get_compile_time_arg_val(3);
// Tiles of each row this core writes, and the cores that share each row.
constexpr uint32_t St = get_compile_time_arg_val(4);
constexpr uint32_t num_slices = get_compile_time_arg_val(5);
constexpr uint32_t partials_sem_id = get_compile_time_arg_val(6);
constexpr bool sliced = num_slices > 1U;

#ifdef COMPUTE_DGAMMA
constexpr bool compute_dgamma = true;
#else
constexpr bool compute_dgamma = false;
#endif

#ifdef GATED
constexpr bool gated = true;
#else
constexpr bool gated = false;
#endif

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t da_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dgamma_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dgate_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);
    // With slicing: this core's slice, then the NOC coordinates of the cores of its row-group in slice order.
    uint32_t slice = 0U;
    uint32_t peer_noc_x[num_slices];
    uint32_t peer_noc_y[num_slices];
    if constexpr (sliced) {
        slice = get_arg_val<uint32_t>(runtime_args_counter++);
        for (uint32_t peer = 0; peer < num_slices; ++peer) {
            peer_noc_x[peer] = get_arg_val<uint32_t>(runtime_args_counter++);
            peer_noc_y[peer] = get_arg_val<uint32_t>(runtime_args_counter++);
        }
    }
    const uint32_t slice_start = slice * St;

    const uint32_t tile_bytes = get_tile_size(cb_dL_da_idx);
    constexpr auto da_args = TensorAccessorArgs<7>();
    constexpr auto dgamma_args = TensorAccessorArgs<da_args.next_compile_time_args_offset()>();
    constexpr auto dgate_args = TensorAccessorArgs<dgamma_args.next_compile_time_args_offset()>();
    const auto da_output_addr_generator = TensorAccessor(da_args, da_output_addr);
    const auto dgamma_output_addr_generator = TensorAccessor(dgamma_args, dgamma_output_addr);
    const auto dgate_output_addr_generator = TensorAccessor(dgate_args, dgate_output_addr);

    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        if constexpr (sliced) {
            all_gather_tile<num_slices>(
                cb_slice_partial_idx,
                cb_gathered_partials_idx,
                slice,
                peer_noc_x,
                peer_noc_y,
                partials_sem_id,
                r - start_row);
        }
        // Write dL_dgate (GATED), dL_da and dL_dgamma_components (COMPUTE_DGAMMA) in blocks. Compute produces the
        // whole row's dL_dgate before the row's first dL_da, so dL_dgate is drained first; dL_da and
        // dL_dgamma_components are interleaved to avoid waiting for the entire Wt tiles at once.
        // rmsnorm_bw() reduces dL_dgamma_components to dL_dgamma with ttnn::sum.
        if constexpr (gated) {
            const uint32_t merged_row_start = head_merged_row<layout_Ht, layout_num_heads>(r) * Wt;
            for (uint32_t c = 0; c < Wt; c += block_size) {
                write_tiles_by_row(
                    cb_dL_dgate_idx,
                    dgate_output_addr_generator,
                    merged_row_start + c,
                    block_size,
                    tile_bytes,
                    block_size);
            }
        }
        for (uint32_t c = 0; c < St; c += block_size) {
            uint32_t start_idx = (r * Wt) + slice_start + c;

            // Write dL_da block
            write_tiles_by_row</* UseBarrier = */ false>(
                cb_dL_da_idx, da_output_addr_generator, start_idx, block_size, tile_bytes, block_size);
            if constexpr (compute_dgamma) {
                // Write dL_dgamma_components block
                write_tiles_by_row</* UseBarrier = */ false>(
                    cb_dL_dgamma_components,
                    dgamma_output_addr_generator,
                    start_idx,
                    block_size,
                    tile_bytes,
                    block_size);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_dL_da_idx, block_size);
            if constexpr (compute_dgamma) {
                cb_pop_front(cb_dL_dgamma_components, block_size);
            }
        }
    }

    if constexpr (sliced) {
        all_gather_tile_finish(partials_sem_id);
    }
}
