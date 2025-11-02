// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// CBs with output data
constexpr uint32_t cb_output_idx = tt::CBIndex::c_6;  // normalized output
constexpr uint32_t cb_mean_idx = tt::CBIndex::c_7;    // mean (for backward pass)
constexpr uint32_t cb_rstd_idx = tt::CBIndex::c_8;    // rstd (for backward pass)

constexpr uint32_t block_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);

#ifdef RETURN_MEAN_RSTD
constexpr bool return_mean_rstd = true;
#else
constexpr bool return_mean_rstd = false;
#endif

constexpr uint32_t onetile = 1;

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
    uint32_t output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t mean_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t rstd_output_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_output_idx);
    constexpr auto output_args = TensorAccessorArgs<2>();
    constexpr auto mean_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto rstd_args = TensorAccessorArgs<mean_args.next_compile_time_args_offset()>();

    const auto output_addr_generator = TensorAccessor(output_args, output_addr, tile_bytes);
    const auto mean_output_addr_generator = TensorAccessor(mean_args, mean_output_addr, tile_bytes);
    const auto rstd_output_addr_generator = TensorAccessor(rstd_args, rstd_output_addr, tile_bytes);

    uint32_t end_row = start_row + num_rows_to_process;
    for (uint32_t r = start_row; r < end_row; ++r) {
        // Write output in blocks
        for (uint32_t c = 0; c < Wt; c += block_size) {
            // Calculate actual number of tiles in this block (handles last block when Wt % block_size != 0)
            const uint32_t current_block_size = (c + block_size > Wt) ? (Wt - c) : block_size;
            uint32_t start_idx = (r * Wt) + c;

            // Write output block
            write_cb_block_to_dram(cb_output_idx, output_addr_generator, start_idx, current_block_size, tile_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_output_idx, current_block_size);
        }

        // Write mean and rstd (one tile per row) if needed
        if constexpr (return_mean_rstd) {
            write_cb_block_to_dram(cb_mean_idx, mean_output_addr_generator, r, 1, tile_bytes);
            write_cb_block_to_dram(cb_rstd_idx, rstd_output_addr_generator, r, 1, tile_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_mean_idx, onetile);
            cb_pop_front(cb_rstd_idx, onetile);
        }
    }
}
