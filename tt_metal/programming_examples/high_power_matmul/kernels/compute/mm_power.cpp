// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

using std::uint32_t;

void kernel_main() {
    uint32_t num_output_tiles = get_arg_val<uint32_t>(0);
    uint32_t Kt = get_arg_val<uint32_t>(1);
    uint32_t num_iterations = get_arg_val<uint32_t>(2);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    // Initialize the matmul operation
    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        if (iter % 10 == 0) {
            // Only print from TRISC0, otherwise we'll get three identical prints
            UNPACK((DPRINT << "Iteration " << iter << " of " << num_iterations << ENDL()));
        }
        for (uint32_t i = 0; i < num_output_tiles; i++) {
            // TRISC1 Acquires the tile registers
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                // TRISC0 waits for the input tiles to be available in the input circular buffers in SRAM (reader kernel
                // populates them)
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);
                // TRISC0 tells the FPU to performs the matrix multiplication
                matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
                // TRISC1 marks the input tiles as used by popping them from the front of the circular buffers
                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_in1, 1);
            }
            // TRISC1 commits the results, transferring ownership of the tile registers to the packer (TRISC2)
            tile_regs_commit();
            // TRISC2 waits for the FPU (TRISC1) to be ready
            tile_regs_wait();
            // TRISC2 reserves space in the output circular buffer for the result tile
            cb_reserve_back(cb_out, 1);
            // TRISC2 tells the packer to pack the result tile into the output circular buffer
            pack_tile(0, cb_out);
            // TRISC2 marks the result tile as used by pushing it to the back of the output circular buffer (writer
            // kernel can read them now)
            cb_push_back(cb_out, 1);
            // TRISC1 releases the tile registers, allowing TRISC0 to acquire them again for the next iteration
            tile_regs_release();
        }
    }
}
