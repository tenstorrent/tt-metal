// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_hw_startup.h"

using std::uint32_t;

// Output block handled per iteration, in tiles. BLOCK_M x BLOCK_N output tiles are accumulated
// in the destination registers simultaneously, so one row of A and one column of B serve
// BLOCK_M x BLOCK_N multiplies instead of one. 1x1 is the original tile-at-a-time behaviour.
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
    uint32_t num_output_blocks = get_arg_val<uint32_t>(0);
    uint32_t Kt = get_arg_val<uint32_t>(1);
    uint32_t num_iterations = get_arg_val<uint32_t>(2);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    // Initialize the matmul operation. compute_kernel_hw_startup configures the unpacker/packer
    // for the input and output CBs (what the old 3-argument mm_init used to cover); matmul_init
    // then sets up the FPU itself.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_init(cb_in0, cb_in1);

    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        if (iter % 10 == 0) {
            // Only print from TRISC0, otherwise we'll get three identical prints
            DPRINT("Iteration {} of {}\n", iter, num_iterations);
        }
        for (uint32_t blk = 0; blk < num_output_blocks; blk++) {
            // TRISC1 acquires the tile registers -- kBlockTiles of them, one accumulator per
            // output tile in the block.
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                // The reader delivers a column slice of A (kBlockM tiles) and a row slice of B
                // (kBlockN tiles) for this step of the shared dimension.
                cb_wait_front(cb_in0, kBlockM);
                cb_wait_front(cb_in1, kBlockN);
                // Every A tile is multiplied against every B tile in the slice, so the two
                // slices are reused kBlockN and kBlockM times respectively before being
                // popped. This is where the DRAM traffic reduction comes from.
                // HIGH_POWER_DISABLE_COMPUTE skips the real FPU work while keeping every CB and
                // tile-register handshake intact, so reader and writer are stimulated exactly as
                // before and neither deadlocks. Output tiles then contain garbage, which is
                // harmless: the app never verifies its output.
#ifndef HIGH_POWER_DISABLE_COMPUTE
                for (uint32_t m = 0; m < kBlockM; m++) {
                    for (uint32_t n = 0; n < kBlockN; n++) {
                        matmul_tiles(cb_in0, cb_in1, m, n, m * kBlockN + n);
                    }
                }
#endif
                cb_pop_front(cb_in0, kBlockM);
                cb_pop_front(cb_in1, kBlockN);
            }
            // TRISC1 commits the results, transferring ownership of the tile registers to the packer (TRISC2)
            tile_regs_commit();
            // TRISC2 waits for the FPU (TRISC1) to be ready
            tile_regs_wait();
            // TRISC2 reserves space in the output circular buffer for the whole block
            cb_reserve_back(cb_out, kBlockTiles);
#ifndef HIGH_POWER_DISABLE_COMPUTE
            for (uint32_t i = 0; i < kBlockTiles; i++) {
                pack_tile(i, cb_out);
            }
#endif
            // TRISC2 marks the result tiles as used by pushing them to the back of the output
            // circular buffer (writer kernel can read them now)
            cb_push_back(cb_out, kBlockTiles);
            // TRISC1 releases the tile registers, allowing TRISC0 to acquire them again
            tile_regs_release();
        }
    }
}
