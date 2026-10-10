// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t num_iterations = get_arg_val<uint32_t>(3);
    // Number of times each output tile is written. 1 = normal. Higher values re-write the same
    // tile to the same address to load the write-side NoC path up to the reader's read volume;
    // the host computes this from HIGH_POWER_WRITE_AMPLIFICATION_PCT.
    uint32_t write_repeats = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(c_args, dst_addr, tile_bytes);

    uint32_t end_id = start_id + num_tiles;

    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        for (uint32_t i = start_id; i < end_id; i++) {
            // When HIGH_POWER_DISABLE_WRITER is set the writer keeps draining the output CB -- so
            // compute never stalls on a full CB -- but issues no DRAM traffic, leaving the output
            // buffer stale. Harmless: the app never verifies its output.
            cb_wait_front(cb_id_out, 1);
#ifndef HIGH_POWER_DISABLE_WRITER
            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
            for (uint32_t rep = 0; rep < write_repeats; rep++) {
                noc_async_write_tile(i, c, l1_read_addr);
                noc_async_write_barrier();
            }
#endif
            cb_pop_front(cb_id_out, 1);
        }
    }
}
