// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

// Which outputs to write. A skipped output has no CB and placeholder accessor args.
#ifdef COMPUTE_GRAD_INPUT
constexpr bool compute_grad_input = true;
#else
constexpr bool compute_grad_input = false;
#endif

#ifdef COMPUTE_GRAD_GAMMA
constexpr bool compute_grad_gamma = true;
#else
constexpr bool compute_grad_gamma = false;
#endif

#ifdef COMPUTE_GRAD_BETA
constexpr bool compute_grad_beta = true;
#else
constexpr bool compute_grad_beta = false;
#endif

// All outputs share a data format; take the tile size from an output CB that exists.
constexpr uint32_t cb_tile_size_idx =
    compute_grad_input ? cb_dx_idx : (compute_grad_gamma ? cb_dgamma_components : cb_dbeta_components);

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t dx_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dgamma_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dbeta_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_tile_size_idx);
    constexpr auto dx_args = TensorAccessorArgs<2>();
    constexpr auto dgamma_args = TensorAccessorArgs<dx_args.next_compile_time_args_offset()>();
    constexpr auto dbeta_args = TensorAccessorArgs<dgamma_args.next_compile_time_args_offset()>();

    const auto dx_output_addr_generator = TensorAccessor(dx_args, dx_output_addr);
    const auto dgamma_output_addr_generator = TensorAccessor(dgamma_args, dgamma_output_addr);
    const auto dbeta_output_addr_generator = TensorAccessor(dbeta_args, dbeta_output_addr);

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
            if constexpr (compute_grad_input) {
                write_tiles_by_row</* UseBarrier = */ false>(
                    cb_dx_idx, dx_output_addr_generator, start_idx, current_block_size, tile_bytes, block_size);
            }
            if constexpr (compute_grad_gamma) {
                write_tiles_by_row</* UseBarrier = */ false>(
                    cb_dgamma_components,
                    dgamma_output_addr_generator,
                    start_idx,
                    current_block_size,
                    tile_bytes,
                    block_size);
            }
            if constexpr (compute_grad_beta) {
                write_tiles_by_row</* UseBarrier = */ false>(
                    cb_dbeta_components,
                    dbeta_output_addr_generator,
                    start_idx,
                    current_block_size,
                    tile_bytes,
                    block_size);
            }
            noc_async_write_barrier();
            if constexpr (compute_grad_input) {
                cb_pop_front(cb_dx_idx, block_size);
            }
            if constexpr (compute_grad_gamma) {
                cb_pop_front(cb_dgamma_components, block_size);
            }
            if constexpr (compute_grad_beta) {
                cb_pop_front(cb_dbeta_components, block_size);
            }
        }
    }
}
