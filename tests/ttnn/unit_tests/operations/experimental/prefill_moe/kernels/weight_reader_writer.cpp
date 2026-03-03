// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Prefill MoE compute V0: Weight reader + output writer (RISCV_0 / Writer)
// Phase 1: Reads weight tiles [K, N_per_core] from interleaved DRAM → CB_WEIGHTS
//          One K-tile's worth of weights (N_per_core tiles) at a time.
// Phase 2: Writes output tiles from CB_OUT → interleaved DRAM.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t output_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_k_tiles = get_arg_val<uint32_t>(2);
    const uint32_t n_per_core = get_arg_val<uint32_t>(3);
    const uint32_t weight_n_total = get_arg_val<uint32_t>(4);  // Total N tiles in weight tensor (180)
    const uint32_t core_n_offset = get_arg_val<uint32_t>(5);   // This core's starting N column tile
    const uint32_t out_start_tile = get_arg_val<uint32_t>(6);  // Output start tile for this core

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
    // positions k * weight_n_total + core_n_offset + n, for n = 0..n_per_core-1
    for (uint32_t k = 0; k < num_k_tiles; ++k) {
        cb_reserve_back(cb_weights, n_per_core);
        uint32_t l1_write_addr = get_write_ptr(cb_weights);

        uint32_t base_tile_id = k * weight_n_total + core_n_offset;
        for (uint32_t n = 0; n < n_per_core; ++n) {
            noc_async_read_page(base_tile_id + n, weight_accessor, l1_write_addr);
            l1_write_addr += weight_page_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_weights, n_per_core);
    }

    // Phase 2: Write output tiles
    cb_wait_front(cb_out, n_per_core);
    uint32_t l1_read_addr = get_read_ptr(cb_out);
    uint32_t out_tile_id = out_start_tile;
    for (uint32_t n = 0; n < n_per_core; ++n) {
        noc_async_write_page(out_tile_id, output_accessor, l1_read_addr);
        l1_read_addr += output_page_bytes;
        out_tile_id++;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, n_per_core);
}
