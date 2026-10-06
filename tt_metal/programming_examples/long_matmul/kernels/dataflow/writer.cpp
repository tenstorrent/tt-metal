// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

// See kernels/compute/compute.cpp. Output arrives one BLOCK_M x BLOCK_N block at a time, in
// row-major order within the block.
#ifndef BLOCK_M
#define BLOCK_M 1
#endif
#ifndef BLOCK_N
#define BLOCK_N 1
#endif

constexpr uint32_t kBlockM = BLOCK_M;
constexpr uint32_t kBlockN = BLOCK_N;
constexpr uint32_t kBlockTiles = kBlockM * kBlockN;

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

    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(c_args, dst_addr, tile_bytes);

    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        for (uint32_t blk = 0; blk < num_output_blocks; blk++) {
            const uint32_t block_id = block_start_id + blk;
            const uint32_t row0 = (block_id / blocks_per_row) * kBlockM;
            const uint32_t col0 = (block_id % blocks_per_row) * kBlockN;

            cb_wait_front(cb_id_out, kBlockTiles);
            uint32_t l1_read_addr = get_read_ptr(cb_id_out);
#ifndef LONG_MATMUL_DISABLE_WRITER
            for (uint32_t m = 0; m < kBlockM; m++) {
                for (uint32_t n = 0; n < kBlockN; n++) {
                    const uint32_t out_tile = (row0 + m) * Nt + (col0 + n);
                    for (uint32_t rep = 0; rep < write_repeats; rep++) {
                        noc_async_write_tile(out_tile, c, l1_read_addr);
                        noc_async_write_barrier();
                    }
                    l1_read_addr += tile_bytes;
                }
            }
#endif
            cb_pop_front(cb_id_out, kBlockTiles);
        }
    }
}
