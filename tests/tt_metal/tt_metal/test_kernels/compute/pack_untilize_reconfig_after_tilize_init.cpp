// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Blackhole: pack_reconfig_data_format between pack_untilize_dest_init and pack_untilize_dest. The pack-untilize
// init programs the dest Z stride the untilize MOP steps from a tile's top face row to its bottom one (two
// faces); the reconfig must leave that stride alone. A reconfig that resets the strides to the Default layout
// (one face) makes the bottom faces of every tile come from the wrong dest rows; one that re-derives them for a
// Tilize layout armed earlier happens to match. With TILIZE_INIT_FIRST the packer is first armed by tilize_init
// and no Default pack init follows, so both reconfig behaviours are covered.
void kernel_main() {
    constexpr std::uint32_t rows = get_arg(args::per_core_block_cnt);
    constexpr std::uint32_t width = get_arg(args::per_core_block_tile_cnt);
    DataflowBuffer input(dfb::in);
    DataflowBuffer output(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);
#if TILIZE_INIT_FIRST
    // Arm the tilize pack layout and leave it armed: only the unpacker is restored for the copies below.
    tilize_init(dfb::in, 1, dfb::out);
    UNPACK((llk_unpack_tilize_uninit(dfb::in)));
#endif
    copy_init(dfb::in);
    pack_untilize_dest_init<width, width>(dfb::out);
    pack_reconfig_data_format(dfb::out);

    for (std::uint32_t row = 0; row < rows; ++row) {
        output.reserve_back(width);
        input.wait_front(width);
        tile_regs_acquire();
        for (std::uint32_t tile = 0; tile < width; ++tile) {
            copy_tile(dfb::in, tile, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_untilize_dest<width, width>(dfb::out, 1, 0);
        tile_regs_release();
        input.pop_front(width);
        output.push_back(width);
    }
    pack_untilize_uninit(dfb::out);
}
