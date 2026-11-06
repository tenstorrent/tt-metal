// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t rms_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_output_idx = tt::CBIndex::c_8;
    constexpr uint32_t cb_rms_output_idx = tt::CBIndex::c_9;
    constexpr uint32_t cb_slice_partial_idx = tt::CBIndex::c_12;
    constexpr uint32_t cb_gathered_partials_idx = tt::CBIndex::c_13;

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);
    // The output's layout for head_merged_row; 1 and 1 when ungated.
    constexpr uint32_t layout_Ht = get_compile_time_arg_val(2);
    constexpr uint32_t layout_num_heads = get_compile_time_arg_val(3);
    // Tiles of each row this core writes, and the cores that share each row.
    constexpr uint32_t St = get_compile_time_arg_val(4);
    constexpr uint32_t num_slices = get_compile_time_arg_val(5);
    constexpr uint32_t partials_sem_id = get_compile_time_arg_val(6);
    constexpr bool sliced = num_slices > 1U;

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

#ifdef RETURN_RMS
    constexpr bool return_rms = true;
#else
    constexpr bool return_rms = false;
#endif

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_output_idx);
    constexpr auto output_args = TensorAccessorArgs<7>();
    constexpr auto rms_output_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    const auto output_addr_generator = TensorAccessor(output_args, output_addr);
    const auto rms_output_addr_generator = TensorAccessor(rms_output_args, rms_output_addr);

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

        if constexpr (return_rms) {
            // Every slice computes the row's rms; slice 0 writes it.
            if (slice == 0U) {
                write_tiles_by_row(cb_rms_output_idx, rms_output_addr_generator, r, onetile, tile_bytes, onetile);
            } else {
                cb_wait_front(cb_rms_output_idx, onetile);
                cb_pop_front(cb_rms_output_idx, onetile);
            }
        }

        const uint32_t output_row = head_merged_row<layout_Ht, layout_num_heads>(r);
        write_full_row_tiles(
            cb_output_idx, output_addr_generator, St, block_size, tile_bytes, output_row * Wt + slice_start);
    }

    if constexpr (sliced) {
        all_gather_tile_finish(partials_sem_id);
    }
}
