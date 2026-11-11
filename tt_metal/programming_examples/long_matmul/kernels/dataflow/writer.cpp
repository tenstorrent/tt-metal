// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "../profile_markers.h"
#include "../config.h"

// Output arrives one BLOCK_M x BLOCK_N block at a time (see ../config.h), in row-major order
// within the block.

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_output_blocks = get_arg_val<uint32_t>(1);
    uint32_t block_start_id = get_arg_val<uint32_t>(2);
    uint32_t num_iterations = get_arg_val<uint32_t>(3);
    // Number of times each output tile is written. 1 = normal. Higher values re-write the same
    // tile to the same address to load the write-side NoC path up to the reader's read volume;
    // the host computes this from LONG_MATMUL_WRITE_AMPLIFICATION_PCT.
    uint32_t write_repeats = get_arg_val<uint32_t>(4);
    uint32_t blocks_per_row = get_arg_val<uint32_t>(5);  // Nt / kBlockN
    uint32_t Nt = get_arg_val<uint32_t>(6);
    uint32_t total_output_blocks = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(c_args, dst_addr, tile_bytes);

    namespace prof = long_matmul_profile;
    DeviceTimestampedData("WRITER_KERNEL_START", /*data=*/0);
    uint64_t wait_cycles = 0;
    uint64_t transfer_cycles = 0;

    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        for (uint32_t blk = 0; blk < num_output_blocks; blk++) {
            // Wrap the same way as the reader, so fixed mode never writes past the end of C.
            const uint32_t block_id = (block_start_id + blk) % total_output_blocks;
            const uint32_t row0 = (block_id / blocks_per_row) * kBlockM;
            const uint32_t col0 = (block_id % blocks_per_row) * kBlockN;

            // When LONG_MATMUL_DISABLE_WRITER is set the writer keeps draining the output CB -- so
            // compute never stalls on a full CB -- but issues no DRAM traffic, leaving the output
            // buffer stale. Harmless: the app never verifies its output.
            uint32_t t = prof::now();
            cb_wait_front(cb_id_out, kBlockTiles);
            prof::add_since(wait_cycles, t);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
#ifndef LONG_MATMUL_DISABLE_WRITER
            t = prof::now();
            for (uint32_t m = 0; m < kBlockM; m++) {
                for (uint32_t n = 0; n < kBlockN; n++) {
                    const uint32_t out_tile = (row0 + m) * Nt + (col0 + n);
                    // Legacy mode waits for every re-write before issuing the next, which makes the
                    // writer, not the reader, set the pace at 100% amplification. Updated mode issues
                    // the re-writes back to back and waits once per tile. With one write per tile
                    // (no amplification) the two are the same.
                    for (uint32_t rep = 0; rep < write_repeats; rep++) {
                        noc_async_write_page(out_tile, c, l1_read_addr);
                        if constexpr (kLegacyMode) {
                            noc_async_write_barrier();
                        }
                    }
                    if constexpr (!kLegacyMode) {
                        noc_async_write_barrier();
                    }
                    l1_read_addr += tile_bytes;
                }
            }
            prof::add_since(transfer_cycles, t);
#endif
            cb_pop_front(cb_id_out, kBlockTiles);
        }
    }

    // The totals go out before KERNEL_END so they fall inside this run's marker window.
    DeviceTimestampedData("WRITER_WAIT_CYCLES", wait_cycles);
    DeviceTimestampedData("WRITER_TRANSFER_CYCLES", transfer_cycles);
    DeviceTimestampedData("WRITER_KERNEL_END", /*data=*/0);
}
