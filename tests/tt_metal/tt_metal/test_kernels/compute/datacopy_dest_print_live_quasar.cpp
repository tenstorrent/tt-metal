// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar LIVE (mid-pipeline) DEST-register dprint validation kernel. Unlike the phase-separated
// datacopy_dest_print_quasar.cpp (copy ALL -> print ALL -> commit -> pack ALL), this reads DEST from
// inside a running, tile-at-a-time pipeline: within each iteration it copies one tile into DEST,
// reads that DEST tile back through dprint_tensix_dest_reg BETWEEN copy_tile and pack_tile, then
// packs. With num_tiles > 1 the pipeline overlaps across iterations (pack of tile N-1 draining while
// unpack streams tile N), so at the read point unpack tile counters are in flight.
//
// This is exactly the pattern that FAULTED before the dbg_halt rendezvous existed: a live DEST read
// mid-pipeline desyncs the unpack tile counter (TILE_COUNTERS hardware fault, error 0x0f00). It now
// passes because dprint_tensix_dest_reg brackets the read with dbg_halt()/dbg_unhalt() -- the mailbox
// rendezvous that quiesces the unpack thread around math's read (pack coordination is deferred). It is
// the positive demonstration that the ported rendezvous enables mid-pipeline DEST inspection.
//
// Loop structure (acquire; wait; copy; print; pack; commit; release) matches the known-good
// bfd_datacopy_quasar.cpp tight loop; only the live DEST read is added.

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

    for (std::uint32_t t = 0; t < num_tiles; ++t) {
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

        tile_regs_acquire();
        tile_regs_wait();
        copy_tile(dfb::in0, 0, 0);

        // Live mid-pipeline DEST read. Word 0 of row 0 equals this tile's host word 0. The rendezvous
        // inside dprint_tensix_dest_reg (dbg_halt/dbg_unhalt) quiesces the unpack thread around the read;
        // without it this same call trips TILE_COUNTERS in the running pipeline.
        dprint_tensix_dest_reg(DataFormat::Float16_b, 0, /*num_rows=*/2);

        pack_tile(0, dfb::out);
        tile_regs_commit();
        tile_regs_release();

        dfb_in.pop_front(1);
        dfb_out.push_back(1);
    }
}
