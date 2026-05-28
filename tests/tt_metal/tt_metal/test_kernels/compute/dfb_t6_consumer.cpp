// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_consumer = get_arg(args::num_entries_per_consumer);
    DataflowBuffer dfb(dfb::in);

    // Each consumer pops exactly num_entries_per_consumer entries from its own TC(s).
    // No modulo-skip is needed: the DFB hardware delivers only this consumer's entries
    // to its TC, so every wait_front/pop_front here is for a tile this consumer owns.
    for (uint32_t tile_id = 0; tile_id < num_entries_per_consumer; tile_id++) {
        dfb.wait_front(1);
        DPRINT_UNPACK("unpack consumer tile id {}\n", tile_id);
        dfb.pop_front(1);
        DPRINT_PACK("pack consumer tile id {}\n", tile_id);
    }
    DPRINT("CBWW\n");
    dfb.finish();
    DPRINT("CBWD\n");
}
