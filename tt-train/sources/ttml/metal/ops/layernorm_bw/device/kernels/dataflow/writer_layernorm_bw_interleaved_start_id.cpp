// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

// CBs with output data
constexpr uint32_t cb_dx_idx = tt::CBIndex::c_9;              // dx (input gradient)
constexpr uint32_t cb_dgamma_components = tt::CBIndex::c_10;  // dgamma components
constexpr uint32_t cb_dbeta_components = tt::CBIndex::c_11;   // dbeta components

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t dx_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dgamma_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dbeta_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_dx_idx);
    const uint32_t tile_bytes_dgamma = get_tile_size(cb_dgamma_components);
    constexpr auto dx_args = TensorAccessorArgs<2>();
    constexpr auto dgamma_args = TensorAccessorArgs<dx_args.next_compile_time_args_offset()>();
    constexpr auto dbeta_args = TensorAccessorArgs<dgamma_args.next_compile_time_args_offset()>();

    const auto dx_output_addr_generator = TensorAccessor(dx_args, dx_output_addr, tile_bytes);
    const auto dgamma_output_addr_generator = TensorAccessor(dgamma_args, dgamma_output_addr, tile_bytes);
    const auto dbeta_output_addr_generator = TensorAccessor(dbeta_args, dbeta_output_addr, tile_bytes);

    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        // Write dx (gradient w.r.t. input), dgamma_components (partial gradients w.r.t. gamma),
        // and dbeta_components (partial gradients w.r.t. beta) in blocks.
        // We interleave the writes to avoid waiting for the entire Wt tiles at once.
        // NOTE: The final dL_dgamma and dL_dbeta (gradients w.r.t. gamma and beta) require
        // reduction across batches, which is performed on the host. Here, we output the
        // per-tile components for host-side reduction.

        for (uint32_t c = 0; c < Wt; c += block_size) {
            // Calculate actual number of tiles in this block (handles last block when Wt % block_size != 0)
            const uint32_t current_block_size = std::min(block_size, Wt - c);
            uint32_t start_idx = (r * Wt) + c;

            // Write dx, dgamma_components, and dbeta_components blocks
            write_tiles_by_row</* UseBarrier = */ false>(
                cb_dx_idx, dx_output_addr_generator, start_idx, current_block_size, tile_bytes, block_size);
            write_tiles_by_row</* UseBarrier = */ false>(
                cb_dgamma_components,
                dgamma_output_addr_generator,
                start_idx,
                current_block_size,
                tile_bytes_dgamma,
                block_size);
            write_tiles_by_row(
                cb_dbeta_components,
                dbeta_output_addr_generator,
                start_idx,
                current_block_size,
                tile_bytes_dgamma,
                block_size);
            // UseBarrier=true above calls noc_async_write_barrier() and pops cb_dbeta_components
            // Must manually pop the first two buffers that used UseBarrier=false
            cb_pop_front(cb_dx_idx, block_size);
            cb_pop_front(cb_dgamma_components, block_size);
        }
    }
}
