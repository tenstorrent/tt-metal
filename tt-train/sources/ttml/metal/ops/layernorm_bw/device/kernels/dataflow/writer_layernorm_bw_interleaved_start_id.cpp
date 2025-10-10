// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

// CBs with output data
constexpr uint32_t cb_dx_idx = tt::CBIndex::c_10;                // dx (input gradient)
constexpr uint32_t cb_dgamma_components = tt::CBIndex::c_11;     // dgamma components
constexpr uint32_t cb_dbeta_components = tt::CBIndex::c_12;      // dbeta components
constexpr uint32_t cb_debug_scaled_sum_idx = tt::CBIndex::c_19;  // DEBUG: scaled dy*gamma sum

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

// constexpr uint32_t onetile = 1;

template <typename AddrGen>
inline void write_cb_block_to_dram(
    uint32_t cb_idx, const AddrGen& addr_gen, uint32_t start_idx, uint32_t block_size, uint32_t tile_bytes) {
    cb_wait_front(cb_idx, block_size);
    uint32_t l1_read_addr = get_read_ptr(cb_idx);
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        noc_async_write_tile(start_idx + block_idx, addr_gen, l1_read_addr);
        l1_read_addr += tile_bytes;
    }
}

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t dx_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dgamma_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dbeta_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_dx_idx);
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
            // DEBUG: Print row number
            if (r == start_row && c == 0) {
                DPRINT << "WRITER: Processing row printing cb_debug_scaled_sum_idx " << r << ENDL();
                cb_wait_front(cb_debug_scaled_sum_idx, 1);
                print_tile(cb_debug_scaled_sum_idx, 0, false);
            }
            uint32_t start_idx = (r * Wt) + c;

            // Write dx block
            write_cb_block_to_dram(cb_dx_idx, dx_output_addr_generator, start_idx, block_size, tile_bytes);

            // Write dgamma_components block
            write_cb_block_to_dram(
                cb_dgamma_components, dgamma_output_addr_generator, start_idx, block_size, tile_bytes);

            // Write dbeta_components block
            write_cb_block_to_dram(cb_dbeta_components, dbeta_output_addr_generator, start_idx, block_size, tile_bytes);

            noc_async_write_barrier();

            cb_pop_front(cb_dx_idx, block_size);
            cb_pop_front(cb_dgamma_components, block_size);
            cb_pop_front(cb_dbeta_components, block_size);
        }
    }
}
