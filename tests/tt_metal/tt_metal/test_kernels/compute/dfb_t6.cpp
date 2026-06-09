// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

// Quick check that DFBAccessor implicitly converts to uint32_t at compile time.
// This is a shim to enable DFB to work with WH/BH LLK compute APIs that expect raw CB ids.
// If implicit conversion (or its constexpr-ness) regressed, this line would fail to compile.
// NOTE: This check is piggybacking along on an unrelated test kernel.
[[maybe_unused]] constexpr uint32_t implicit_id = dfb::in;

void kernel_main() {
    constexpr uint32_t num_entries = get_arg(args::num_entries);
    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    for (uint32_t tile_id = 0; tile_id < num_entries; tile_id++) {
        // DPRINT("rbw\n");
        dfb_in.wait_front(1);
        // do some processing on unpacker
        dfb_in.pop_front(1);

        // add a sync here
        // shouldn't reserve back until data is ready to be consumed here...

        dfb_out.reserve_back(1);
        // do some processing on packer
        dfb_out.push_back(1);
        // DPRINT("pbd\n");
    }
    DPRINT("PFW\n");
    dfb_out.finish();
    DPRINT("PFD\n");
}
