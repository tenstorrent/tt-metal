// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Writes this core's receiver info record (stream.py): receiver index, sender NOC x, y, index in the sender's group.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    volatile tt_l1_ptr uint32_t* info = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    // receiver index, sender NOC x, y, index in its group, then the span positions (stream.py span_ns)
    for (uint32_t i = 0; i < 16; ++i) {
        info[i] = i < NARGS ? get_arg_val<uint32_t>(1 + i) : 0;
    }
}
