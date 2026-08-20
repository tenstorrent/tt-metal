// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reports this node's L1 addresses for `recv` and `stage` into the recv entry, so the host can
// compare them across nodes. "Lean" node set: {recv, stage}.
//
// Every cross-core kernel in this directory assumes a DFB lands at the SAME L1 address on the peer
// node as it does locally, because that local address is the only one it can name. This kernel
// exists to check that assumption on nodes whose resource sets differ.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/scratchpad.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    DataflowBuffer dfb_recv(dfb::recv);
    Scratchpad<uint32_t> stage(scratch::stage);

    dfb_recv.reserve_back(1);
    CoreLocalMem<volatile uint32_t> slot(dfb_recv.get_write_ptr());
    slot[0] = dfb_recv.get_write_ptr();
    slot[1] = stage.get_base_address();
    slot[2] = 0xdeadbeefu;  // "lean" marker
    dfb_recv.push_back(1);
}
