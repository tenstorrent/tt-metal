// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Copies one DFB to another where a DFB entry holds MORE THAN ONE TILE.
//
// Every other Quasar compute test runs with entry_size == one tile, which hides the
// entry->tile index conversion on both sides of DEST: when there is one tile per entry an
// entry index and a tile index are numerically the same, so a missing conversion cannot be
// observed. With tiles_per_entry > 1 the two diverge and the conversion is exercised:
//
//   unpack: copy_tile(in, t, t) must read tile t of the CURRENT entry, i.e.
//           rd_entry_idx scaled into tiles plus t (llk_unpack_A).
//   pack:   pack_tile(t, out) must write tile t of the CURRENT entry, i.e.
//           (wr_entry_idx * tiles_per_entry) + t (get_output_tile_index, #56194).
//
// One entry is handled per DEST acquire: wait for one input entry, unpack its
// tiles_per_entry tiles into DEST tiles [0, tiles_per_entry), pack them back out into the
// one reserved output entry, then release. Packing is in-order (no out_of_order_output), so
// the packer's wr_entry_ptr walks the tiles inside the entry by itself; that is the path a
// real multi-tile-entry op uses.
//
// tiles_per_entry must not exceed the DEST section capacity for the sync mode and DEST width.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr std::uint32_t num_entries = get_arg(args::num_entries);
    constexpr std::uint32_t tiles_per_entry = get_arg(args::tiles_per_entry);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    copy_init(dfb::in);

    for (std::uint32_t e = 0; e < num_entries; ++e) {
        tile_regs_acquire();
        tile_regs_wait();

        // One entry in, one entry out. The counts here are ENTRIES, not tiles.
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

        // Tile t of the input entry -> DEST tile t -> tile t of the output entry.
        for (std::uint32_t t = 0; t < tiles_per_entry; ++t) {
            copy_tile(dfb::in, t, t);
        }
        for (std::uint32_t t = 0; t < tiles_per_entry; ++t) {
            pack_tile(t, dfb::out);
        }

        dfb_in.pop_front(1);
        dfb_out.push_back(1);

        tile_regs_commit();
        tile_regs_release();
    }
}
