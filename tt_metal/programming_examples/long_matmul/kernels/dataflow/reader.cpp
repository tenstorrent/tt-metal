// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
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
    uint32_t Kt = get_arg(args::Kt);
    uint32_t Nt = get_arg(args::Nt);
    uint32_t block_start_id = get_arg(args::block_start_id);
    uint32_t num_output_blocks = get_arg(args::num_output_blocks);
    uint32_t num_iterations = get_arg(args::num_iterations);
    uint32_t blocks_per_row = get_arg(args::blocks_per_row);  // Nt / kBlockN
    uint32_t total_output_blocks = get_arg(args::total_output_blocks);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    const uint32_t in0_tile_bytes = dfb_in0.get_entry_size();
    const uint32_t in1_tile_bytes = dfb_in1.get_entry_size();

    // A and B are tensors bound to this kernel by name in the host code; their page size is one tile.
    const auto a = TensorAccessor(tensor::a);
    const auto b = TensorAccessor(tensor::b);

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
                // When LONG_MATMUL_DISABLE_READER is set the reader keeps its full DFB handshake --
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
                    dfb_in0.reserve_back(kBlockM);
                    prof::add_since(wait_cycles, t);
                    uint32_t l1_addr = dfb_in0.get_write_ptr();
                    if (read) {
                        t = prof::now();
                        for (uint32_t m = 0; m < kBlockM; m++) {
                            noc_async_read_page((row0 + m) * Kt + k, a, l1_addr);
                            l1_addr += in0_tile_bytes;
                        }
                        noc_async_read_barrier();
                        prof::add_since(transfer_cycles, t);
                    }
                    dfb_in0.push_back(kBlockM);
                }
                {
                    // Row slice of B: inner index k, columns col0..col0+kBlockN-1.
                    uint32_t t = prof::now();
                    dfb_in1.reserve_back(kBlockN);
                    prof::add_since(wait_cycles, t);
                    uint32_t l1_addr = dfb_in1.get_write_ptr();
                    if (read) {
                        t = prof::now();
                        for (uint32_t n = 0; n < kBlockN; n++) {
                            noc_async_read_page(k * Nt + (col0 + n), b, l1_addr);
                            l1_addr += in1_tile_bytes;
                        }
                        noc_async_read_barrier();
                        prof::add_since(transfer_cycles, t);
                    }
                    dfb_in1.push_back(kBlockN);
                }
            }
        }
    }

    // The totals go out before KERNEL_END so they fall inside this run's marker window.
    DeviceTimestampedData("READER_WAIT_CYCLES", wait_cycles);
    DeviceTimestampedData("READER_TRANSFER_CYCLES", transfer_cycles);
    DeviceTimestampedData("READER_KERNEL_END", /*data=*/0);
}
