// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Same report as addr_report_lean.cpp, but this node's set also holds the `pad` DFB, which is
// declared BEFORE `recv` in the ProgramSpec. If DFB L1 addresses are assigned per node from the
// node's own resource set, `recv` and `stage` land somewhere else here than on the lean node.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/scratchpad.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    DataflowBuffer dfb_pad(dfb::pad);
    DataflowBuffer dfb_recv(dfb::recv);
    Scratchpad<uint32_t> stage(scratch::stage);

    dfb_pad.reserve_back(1);
    dfb_pad.push_back(1);

    dfb_recv.reserve_back(1);
    CoreLocalMem<volatile uint32_t> slot(dfb_recv.get_write_ptr());
    slot[0] = dfb_recv.get_write_ptr();
    slot[1] = stage.get_base_address();
    slot[2] = 0xfeedface;  // "fat" marker
    slot[3] = dfb_pad.get_write_ptr();
    dfb_recv.push_back(1);
}
