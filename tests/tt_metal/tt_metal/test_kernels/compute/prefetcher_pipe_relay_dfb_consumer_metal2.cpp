// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 TRISC consumer of a DataflowBuffer laid over PrefetcherPipe rings.
//
// One DFB entry is one delivered entry, so this consumer addresses the tiles inside an entry the
// way the 1D matmul's compute kernel does: take the entry's read pointer once, then step by the
// tile size. Firmware resets the DFB pointers every launch while the pipe cursor is durable; the
// generated dfb::relay binding is a relay token, so constructing the DataflowBuffer re-aligns to
// the pipe's checkpoint before the first wait -- finding this core's pipe from the relay id.
//
// It records the first word of every tile it reads so the host can check that compute saw each
// delivered tile exactly once, in order.
//
// Compile-time args (named):
//   total_entries, tiles_per_entry, tile_bytes,
//   result_addr - L1 address of [entries_consumed, first word of each tile ...]

#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
#ifdef UCK_CHLKC_UNPACK
    constexpr uint32_t total_entries = get_arg(args::total_entries);
    constexpr uint32_t tiles_per_entry = get_arg(args::tiles_per_entry);
    constexpr uint32_t tile_bytes = get_arg(args::tile_bytes);
    constexpr uint32_t result_addr = get_arg(args::result_addr);

    DataflowBuffer relay(dfb::relay);
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);

    uint32_t entries_consumed = 0;
    for (uint32_t entry = 0; entry < total_entries; ++entry) {
        relay.wait_front(1);
        const uint32_t entry_base = relay.get_read_ptr() << cb_addr_shift;
        for (uint32_t tile = 0; tile < tiles_per_entry; ++tile) {
            result[1 + entry * tiles_per_entry + tile] =
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(entry_base + tile * tile_bytes);
        }
        relay.pop_front(1);
        ++entries_consumed;
    }
    result[0] = entries_consumed;
#endif
}
