// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

// See kernels/compute/mm_power.cpp. With a BLOCK_M x BLOCK_N output block, one column slice of
// A and one row slice of B feed BLOCK_M * BLOCK_N multiplies, so per multiply this reads
// (BLOCK_M + BLOCK_N) / (BLOCK_M * BLOCK_N) tiles instead of 2.
#ifndef BLOCK_M
#define BLOCK_M 1
#endif
#ifndef BLOCK_N
#define BLOCK_N 1
#endif

constexpr uint32_t kBlockM = BLOCK_M;
constexpr uint32_t kBlockN = BLOCK_N;

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t block_start_id = get_arg_val<uint32_t>(5);
    uint32_t num_output_blocks = get_arg_val<uint32_t>(6);
    uint32_t num_iterations = get_arg_val<uint32_t>(7);
    uint32_t blocks_per_row = get_arg_val<uint32_t>(8);  // Nt / kBlockN

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, src0_addr, in0_tile_bytes);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, src1_addr, in1_tile_bytes);

    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        for (uint32_t blk = 0; blk < num_output_blocks; blk++) {
            const uint32_t block_id = block_start_id + blk;
            const uint32_t block_row = block_id / blocks_per_row;
            const uint32_t block_col = block_id % blocks_per_row;
            const uint32_t row0 = block_row * kBlockM;  // first output tile row in the block
            const uint32_t col0 = block_col * kBlockN;  // first output tile column in the block

            for (uint32_t k = 0; k < Kt; k++) {
                {
                    // Column slice of A: rows row0..row0+kBlockM-1 at inner index k.
                    cb_reserve_back(cb_id_in0, kBlockM);
                    uint32_t l1_addr = get_write_ptr(cb_id_in0);
#ifndef HIGH_POWER_DISABLE_READER
                    for (uint32_t m = 0; m < kBlockM; m++) {
                        noc_async_read_tile((row0 + m) * Kt + k, a, l1_addr);
                        l1_addr += in0_tile_bytes;
                    }
                    noc_async_read_barrier();
#endif
                    cb_push_back(cb_id_in0, kBlockM);
                }
                {
                    // Row slice of B: inner index k, columns col0..col0+kBlockN-1.
                    cb_reserve_back(cb_id_in1, kBlockN);
                    uint32_t l1_addr = get_write_ptr(cb_id_in1);
#ifndef HIGH_POWER_DISABLE_READER
                    for (uint32_t n = 0; n < kBlockN; n++) {
                        noc_async_read_tile(k * Nt + (col0 + n), b, l1_addr);
                        l1_addr += in1_tile_bytes;
                    }
                    noc_async_read_barrier();
#endif
                    cb_push_back(cb_id_in1, kBlockN);
                }
            }
        }
    }
}
