// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Int32 / UInt32 copy with no unpack_modes entry: the host derives the DEST route from the operand format alone.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr std::uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    copy_init(dfb::in);

    for (std::uint32_t t = 0; t < num_tiles; ++t) {
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);
        tile_regs_acquire();
        copy_tile(dfb::in, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::out);
        tile_regs_release();
        dfb_in.pop_front(1);
        dfb_out.push_back(1);
    }
}
