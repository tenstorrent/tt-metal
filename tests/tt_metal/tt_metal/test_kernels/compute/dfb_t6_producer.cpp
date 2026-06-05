// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);
    DataflowBuffer dfb(dfb::out);

    for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id++) {
        DPRINT("producer tile id {}\n", tile_id);
        dfb.reserve_back(1);
        dfb.push_back(1);
    }
    DPRINT("PFW\n");
    dfb.finish();
    DPRINT("PFD\n");
}
