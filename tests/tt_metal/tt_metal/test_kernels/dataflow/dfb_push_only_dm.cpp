// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Posts `num_entries` entries to every tile counter dfb::out round-robins over and exits without waiting for them to
// be consumed, deliberately leaving those counters with posted != acked after the program completes. The posts go
// straight to the counters rather than through a DataflowBuffer, whose destructor would wait for them to be consumed.

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries = get_arg(args::num_entries);

    const LocalDFBInterface& dfb = get_local_dfb_interface(dfb::out);
    for (uint8_t i = 0; i < dfb.num_tcs_to_rr; i++) {
        const dfb::PackedTileCounter ptc = dfb.tc_slots[i].packed_tile_counter;
        overlay::llk_intf_inc_posted(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc), num_entries);
    }
    // The posts are posted writes: read a counter back so they have landed before the kernel reports done.
    const dfb::PackedTileCounter last = dfb.tc_slots[dfb.num_tcs_to_rr - 1].packed_tile_counter;
    (void)overlay::llk_intf_get_occupancy(dfb::get_tensix_id(last), dfb::get_counter_id(last));
}
