// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Prefill MoE compute: Weight reader + SwiGLU output writer (RISCV_0 / Writer)
// Phase 1: Reads weight tiles [K, N_weight] from interleaved DRAM → CB_WEIGHTS
//          One K-tile's worth of weights (n_weight_tiles tiles) at a time.
// Phase 2: Writes SwiGLU output tiles (n_output_tiles) from CB_OUT → interleaved DRAM.
//
// Unlike weight_reader_writer.cpp, this kernel handles different counts for
// weight tiles (e.g., 12 = 6 gate + 6 up) and output tiles (e.g., 6 SwiGLU results).

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_k_tiles = get_arg_val<uint32_t>(2);
    const uint32_t n_weight_tiles = get_arg_val<uint32_t>(3);      // Weight tiles per core per K (12)
    const uint32_t weight_n_total = get_arg_val<uint32_t>(4);      // Total N tiles in weight tensor (180)
    const uint32_t core_weight_offset = get_arg_val<uint32_t>(5);  // This core's starting N column tile in weight
    const uint32_t out_start_tile = get_arg_val<uint32_t>(6);      // Output start tile for this core
    const uint32_t n_output_tiles = get_arg_val<uint32_t>(7);      // Output tiles per core (6)

    // Compile-time args: [TensorAccessorArgs(weight), TensorAccessorArgs(output)]
    constexpr auto weight_args = TensorAccessorArgs<0>();
    constexpr auto output_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_weights = 1;
    constexpr uint32_t cb_out = 2;

    const uint32_t weight_page_bytes = get_local_cb_interface(cb_weights).fifo_page_size;
    const uint32_t output_page_bytes = get_local_cb_interface(cb_out).fifo_page_size;

    const auto weight_accessor = TensorAccessor(weight_args, weight_addr, weight_page_bytes);
    const auto output_accessor = TensorAccessor(output_args, output_addr, output_page_bytes);

    // Phase 1: Read weight tiles, one K-row at a time
    // Weight tensor is [K, N] tiles. For K-row k, core reads tiles at
    // positions k * weight_n_total + core_weight_offset + n, for n = 0..n_weight_tiles-1
    for (uint32_t k = 0; k < num_k_tiles; ++k) {
        cb_reserve_back(cb_weights, n_weight_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_weights);

        uint32_t base_tile_id = k * weight_n_total + core_weight_offset;
        for (uint32_t n = 0; n < n_weight_tiles; ++n) {
            noc_async_read_page(base_tile_id + n, weight_accessor, l1_write_addr);
            l1_write_addr += weight_page_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_weights, n_weight_tiles);
    }

    // Phase 2: Write SwiGLU output tiles
    cb_wait_front(cb_out, n_output_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb_out);
    uint32_t out_tile_id = out_start_tile;
    for (uint32_t n = 0; n < n_output_tiles; ++n) {
        noc_async_write_page(out_tile_id, output_accessor, l1_read_addr);
        l1_read_addr += output_page_bytes;
        out_tile_id++;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, n_output_tiles);
}
