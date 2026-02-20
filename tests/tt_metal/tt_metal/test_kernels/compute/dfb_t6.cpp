// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/debug/dprint.h"

void kernel_main() {
    const uint32_t num_entries = get_compile_time_arg_val(0);

    uint32_t logical_dfb_in = get_arg_val<uint32_t>(0);
    uint32_t logical_dfb_out = get_arg_val<uint32_t>(1);

    experimental::DataflowBuffer dfb_in(logical_dfb_in);
    experimental::DataflowBuffer dfb_out(logical_dfb_out);

    for (uint32_t tile_id = 0; tile_id < num_entries; tile_id++) {
        // DPRINT << "rbw" << ENDL();
        dfb_in.wait_front(1);
        // do some processing on unpacker
        dfb_in.pop_front(1);

        // add a sync here
        // shouldn't reserve back until data is ready to be consumed here...

        dfb_out.reserve_back(1);
        // do some processing on packer
        dfb_out.push_back(1);
        // DPRINT << "pbd" << ENDL();
    }
    DPRINT << "PFW" << ENDL();
    dfb_out.finish();
    DPRINT << "PFD" << ENDL();
}
