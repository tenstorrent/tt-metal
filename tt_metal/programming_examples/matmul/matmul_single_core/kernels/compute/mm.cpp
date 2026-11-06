// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "hostdevcommon/kernel_structs.h"

using std::uint32_t;

/**
 * @brief Main kernel function for batched matrix multiplication (BMM).
 *
 * This function performs a blocked outer product matrix multiplication using tiles.
 * It initializes the matrix engine (FPU) and sets up dataflow buffers for input and output.
 * For each output tile (indexed by mt and nt), it:
 *   - Acquires the destination buffer.
 *   - Iterates over the K dimension (kt), waiting for input tiles to be available in the dataflow buffers.
 *   - Performs a tile-wise matrix multiplication using `matmul_tiles`.
 *   - Pops the used tiles from the input buffers.
 *   - After processing all K tiles, reserves space in the output buffer, packs the result tile, and pushes it to the
 * output buffer.
 *   - Releases the destination buffer.
 *
 * Compile-time arguments:
 *   - Mt: Number of output tile rows.
 *   - Kt: Number of tiles in the reduction dimension.
 *   - Nt: Number of output tile columns.
 *
 * Dataflow buffers:
 *   - dfb::in0: Input buffer for matrix A tiles.
 *   - dfb::in1: Input buffer for matrix B tiles.
 *   - dfb::out0: Output buffer for result tiles.
 *
 * Assumes that input tiles are provided in the correct order and that the reader is responsible for supplying
 * the appropriate tiles for each output tile computation.
 */
void kernel_main() {
    const uint32_t Mt = get_arg(args::Mt);
    const uint32_t Kt = get_arg(args::Kt);
    const uint32_t Nt = get_arg(args::Nt);
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out0);

    // Setup the FPU (matrix engine) for the matmul operation. And specify the input
    // and output dataflow buffers.
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::out0);
    matmul_init(dfb::in0, dfb::in1);

    // the simplest possible version of outer product blocked matmul
    // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            // Make sure registers can be used for the output tile. This also sets the registers to zero.
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                // Wait for the input tiles to be available in the input dataflow buffers.
                dfb_in0.wait_front(1);
                dfb_in1.wait_front(1);

                // Perform the matrix multiplication for the current tile.
                // NOTE: This function also accumulates the result into the destination tile.
                matmul_tiles(dfb::in0, dfb::in1, 0, 0, 0);

                // Mark the input tiles as used by popping them from the front of the dataflow buffers.
                dfb_in0.pop_front(1);
                dfb_in1.pop_front(1);
            }

            // Commit and wait for the registers are populated with the results from the FPU
            tile_regs_commit();
            tile_regs_wait();

            // Ensure the output dataflow buffer has space for the result tile.
            dfb_out.reserve_back(1);
            // Pack the result tile into the output dataflow buffer.
            pack_tile(0, dfb::out0);
            // Mark the output tile as ready so the writer can read it.
            dfb_out.push_back(1);

            // We don't need the registers anymore, so we can release them and prepare for the next output tile.
            tile_regs_release();
        }
    }
}
