// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "hostdevcommon/kernel_structs.h"

using std::uint32_t;

namespace NAMESPACE {
/**
 * @brief Main kernel function for batched matrix multiplication (BMM).
 *
 * This function performs a blocked outer product matrix multiplication using tiles.
 * It initializes the matrix engine (FPU) and sets up circular buffers for input and output.
 * For each output tile (indexed by mt and nt), it:
 *   - Acquires the destination buffer.
 *   - Iterates over the K dimension (kt), waiting for input tiles to be available in the circular buffers.
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
 * Circular buffers:
 *   - cb_in0: Input buffer for matrix A tiles.
 *   - cb_in1: Input buffer for matrix B tiles.
 *   - cb_out: Output buffer for result tiles.
 *
 * Assumes that input tiles are provided in the correct order and that the reader is responsible for supplying
 * the appropriate tiles for each output tile computation.
 */
void MAIN {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    const uint32_t Nt = get_compile_time_arg_val(2);
    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    // Setup the FPU (matrix engine) for the matmul operation. And specify the input
    // and output circular buffers.
    mm_init(cb_in0, cb_in1, cb_out);

    // the simplest possible version of outer product blocked matmul
    // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            // Make sure registers can be used for the output tile. This also sets the registers to zero.
            acquire_dst();
            for (uint32_t kt = 0; kt < Kt; kt++) {
                // Wait for the input tiles to be available in the input circular buffers.
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);

                // Perform the matrix multiplication for the current tile.
                // NOTE: This function also accumulates the result into the destination tile.
                matmul_tiles(cb_in0, cb_in1, 0, 0, 0, false);

                // Mark the input tiles as used by popping them from the front of the circular buffers.
                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_in1, 1);
            }

            // Ensure the output circular buffer has space for the result tile.
            cb_reserve_back(cb_out, 1);
            // Pack the result tile into the output circular buffer.
            pack_tile(0, cb_out);
            // Mark the output tile as ready so the writer can read it.
            cb_push_back(cb_out, 1);

            // We don't need the registers anymore, so we can release them and prepare for the next output tile.
            release_dst();
        }
    }
}
}  // namespace NAMESPACE
