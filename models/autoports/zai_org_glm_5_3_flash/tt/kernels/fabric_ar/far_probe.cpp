// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 14 fabric all-reduce, "cb" mode setup probe (tt/fabric_ar.py FabricPhase.probe): the lane cores' landing /
// own / result buffers are circular buffers of the program (on the lane cores only, so they do not reserve L1 on the
// other cores the way L1 tensors do); this one-off program, with exactly the same lane CB list, writes each CB's base
// address into the persistent sync words CB_ADDR + i, read back by the host and passed to producers / consumers /
// peers. Runtime args: sync address, n, then n CB indices.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "far_common.hpp"

void kernel_main() {
    const uint32_t sync = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    for (uint32_t i = 0; i < n; i++) {
        *far::word(sync, far::CB_ADDR + i) = get_write_ptr(get_arg_val<uint32_t>(2 + i));
    }
}
