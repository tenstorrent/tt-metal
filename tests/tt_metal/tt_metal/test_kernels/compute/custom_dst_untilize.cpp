// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/experimental/custom_pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr std::uint32_t rows = get_arg(args::per_core_block_cnt);
    constexpr std::uint32_t full_width = get_arg(args::per_core_block_tile_cnt);
    constexpr std::uint32_t block_width = full_width == 1 ? 1 : 4;
    static_assert(full_width % block_width == 0);
    DataflowBuffer input(dfb::in);
    DataflowBuffer output(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    copy_init(dfb::in);
    // Set up the caller's MATH remap before copying. The custom init below
    // runs with populated DEST and must preserve that state and its semaphore.
    pack_untilize_dest_init<block_width, full_width>(dfb::out);

    for (std::uint32_t row = 0; row < rows; ++row) {
        output.reserve_back(full_width);
        for (std::uint32_t block = 0; block < full_width / block_width; ++block) {
            input.wait_front(block_width);
            tile_regs_acquire();
            for (std::uint32_t tile = 0; tile < block_width; ++tile) {
                copy_tile(dfb::in, tile, tile);
            }
            tile_regs_commit();
            tile_regs_wait();
            custom_pack_untilize_dest_init<block_width, full_width>(dfb::out, EXPLICIT_FACE_R_DIM, EXPLICIT_NUM_FACES);
            custom_pack_untilize_dest<block_width, full_width>(
                dfb::out, EXPLICIT_FACE_R_DIM, EXPLICIT_NUM_FACES, 1, block);
            tile_regs_release();
            input.pop_front(block_width);
        }
        output.push_back(full_width);
    }
    pack_untilize_uninit(dfb::out);
}
