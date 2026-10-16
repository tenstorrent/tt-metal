// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "../profile_markers.h"
#include "../config.h"

// With a BLOCK_M x BLOCK_N output block (see ../config.h), one column slice of A and one row
// slice of B feed BLOCK_M * BLOCK_N multiplies, so per multiply this reads
// (BLOCK_M + BLOCK_N) / (BLOCK_M * BLOCK_N) tiles instead of 2.

#if defined(LONG_MATMUL_DISABLE_READER)
constexpr bool kReaderIdle = true;
#else
constexpr bool kReaderIdle = false;
#endif

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t Kt = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(3);
    uint32_t block_start_id = get_arg_val<uint32_t>(4);
    uint32_t num_output_blocks = get_arg_val<uint32_t>(5);
    uint32_t num_iterations = get_arg_val<uint32_t>(6);
    uint32_t blocks_per_row = get_arg_val<uint32_t>(7);  // Nt / kBlockN
    uint32_t total_output_blocks = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, src0_addr, in0_tile_bytes);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, src1_addr, in1_tile_bytes);

    namespace prof = long_matmul_profile;
    DeviceTimestampedData("READER_KERNEL_START", /*data=*/0);
    uint64_t wait_cycles = 0;
    uint64_t transfer_cycles = 0;
    [[maybe_unused]] uint32_t real_steps_left = kRealStepsWhenIdle;  // only used when the reader is idle

    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        for (uint32_t blk = 0; blk < num_output_blocks; blk++) {
            // Wrap so that in fixed mode a core whose range runs past the last block starts
            // over at block 0 instead of reading past the end of A and B.
            const uint32_t block_id = (block_start_id + blk) % total_output_blocks;
            const uint32_t block_row = block_id / blocks_per_row;
            const uint32_t block_col = block_id % blocks_per_row;
            const uint32_t row0 = block_row * kBlockM;  // first output tile row in the block
            const uint32_t col0 = block_col * kBlockN;  // first output tile column in the block

            for (uint32_t k = 0; k < Kt; k++) {
                // When LONG_MATMUL_DISABLE_READER is set the reader keeps its full CB handshake --
                // so compute still gets its tiles "delivered" every step and never deadlocks --
                // but issues no DRAM traffic. In updated mode the first kRealStepsWhenIdle steps
                // still read their slices for real, so compute works on real operands; in legacy
                // mode it works on whatever L1 already held.
                bool read = true;
                if constexpr (kReaderIdle) {
                    read = real_steps_left > 0;
                    real_steps_left -= read ? 1 : 0;
                }
                {
                    // Column slice of A: rows row0..row0+kBlockM-1 at inner index k.
                    uint32_t t = prof::now();
                    cb_reserve_back(cb_id_in0, kBlockM);
                    prof::add_since(wait_cycles, t);
                    uint32_t l1_addr = get_write_ptr(cb_id_in0);
                    if (read) {
                        t = prof::now();
                        for (uint32_t m = 0; m < kBlockM; m++) {
                            noc_async_read_page((row0 + m) * Kt + k, a, l1_addr);
                            l1_addr += in0_tile_bytes;
                        }
                        noc_async_read_barrier();
                        prof::add_since(transfer_cycles, t);
                    }
                    cb_push_back(cb_id_in0, kBlockM);
                }
                {
                    // Row slice of B: inner index k, columns col0..col0+kBlockN-1.
                    uint32_t t = prof::now();
                    cb_reserve_back(cb_id_in1, kBlockN);
                    prof::add_since(wait_cycles, t);
                    uint32_t l1_addr = get_write_ptr(cb_id_in1);
                    if (read) {
                        t = prof::now();
                        for (uint32_t n = 0; n < kBlockN; n++) {
                            noc_async_read_page(k * Nt + (col0 + n), b, l1_addr);
                            l1_addr += in1_tile_bytes;
                        }
                        noc_async_read_barrier();
                        prof::add_since(transfer_cycles, t);
                    }
                    cb_push_back(cb_id_in1, kBlockN);
                }
            }
        }
    }

    // The totals go out before KERNEL_END so they fall inside this run's marker window.
    DeviceTimestampedData("READER_WAIT_CYCLES", wait_cycles);
    DeviceTimestampedData("READER_TRANSFER_CYCLES", transfer_cycles);
    DeviceTimestampedData("READER_KERNEL_END", /*data=*/0);
}
