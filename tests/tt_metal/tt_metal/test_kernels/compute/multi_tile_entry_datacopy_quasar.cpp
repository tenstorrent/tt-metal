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
// Entries are waited for and reserved entries_per_batch (2) at a time, so the tile index
// passed to copy_tile runs across the whole batch: tiles of the second entry exercise the
// tile_index / tiles_per_entry step that a one-entry batch never reaches. Each entry still
// gets its own DEST acquire: unpack its tiles_per_entry tiles into DEST tiles
// [0, tiles_per_entry), pack them back out, release. Packing is in-order (no
// out_of_order_output), so the packer's wr_entry_ptr walks the batch by itself; that is the
// path a real multi-tile-entry op uses.
//
// tiles_per_entry must not exceed the DEST section capacity for the sync mode and DEST width,
// and num_entries must be a multiple of entries_per_batch.

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
    constexpr std::uint32_t entries_per_batch = 2;
    static_assert(num_entries % entries_per_batch == 0, "num_entries must be a multiple of entries_per_batch");

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    copy_init(dfb::in);

    for (std::uint32_t b = 0; b < num_entries / entries_per_batch; ++b) {
        // Two entries in, two entries out. The counts here are ENTRIES, not tiles.
        dfb_in.wait_front(entries_per_batch);
        dfb_out.reserve_back(entries_per_batch);

        for (std::uint32_t e = 0; e < entries_per_batch; ++e) {
            tile_regs_acquire();
            tile_regs_wait();

            // Tile t of entry e -> DEST tile t -> tile t of output entry e. The input index is
            // the tile's position in the whole batch, so entry 1 is reached through it.
            for (std::uint32_t t = 0; t < tiles_per_entry; ++t) {
                copy_tile(dfb::in, e * tiles_per_entry + t, t);
            }
            for (std::uint32_t t = 0; t < tiles_per_entry; ++t) {
                pack_tile(t, dfb::out);
            }

            tile_regs_commit();
            tile_regs_release();
        }

        dfb_in.pop_front(entries_per_batch);
        dfb_out.push_back(entries_per_batch);
    }
}
