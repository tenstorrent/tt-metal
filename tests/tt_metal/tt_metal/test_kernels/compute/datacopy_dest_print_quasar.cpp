// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar DEST-register dprint validation kernel. Copies num_tiles tiles from one input DFB into
// one output DFB and, in between, reads each copied tile back out of the DEST register through the
// RISC memory-mapped window (dprint_tensix_dest_reg) and prints it, so the host can confirm DEST
// holds what it placed.
//
// This deliberately follows the canonical, phase-separated dprint-dest pattern
// (tests/.../compute/eltwise_copy_print_dest.cpp): a single tile_regs_acquire, copy ALL tiles into
// distinct DEST registers, read ALL, then tile_regs_commit -> tile_regs_wait -> pack ALL -> release.
// dprint_tensix_dest_reg brackets its read with the unpack<->math dbg_halt/dbg_unhalt rendezvous, but
// this kernel is also structurally quiesced (the read sits in the copy->read window before any pack),
// so it is the simplest safe shape; the live-pipeline variant is exercised separately.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/dprint.h"
#include "api/debug/dprint_tensix.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr std::uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer dfb_in(dfb::in0);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in0, dfb::out);
    copy_init(dfb::in0);

    dfb_in.wait_front(num_tiles);
    dfb_out.reserve_back(num_tiles);

    tile_regs_acquire();
    for (std::uint32_t b = 0; b < num_tiles; ++b) {
        copy_tile(dfb::in0, b, b);
    }
    // Quiesced DEST read: all tiles are in DEST and nothing is packing yet. Print the first 2 rows
    // per tile (row 0 word 0 == the host tile's word 0) to keep the emulated print stream short.
    for (std::uint32_t b = 0; b < num_tiles; ++b) {
        dprint_tensix_dest_reg(DataFormat::Float16_b, b, /*num_rows=*/2);
    }
    tile_regs_commit();

    tile_regs_wait();
    for (std::uint32_t b = 0; b < num_tiles; ++b) {
        pack_tile(b, dfb::out);
    }
    tile_regs_release();

    dfb_in.pop_front(num_tiles);
    dfb_out.push_back(num_tiles);
}
