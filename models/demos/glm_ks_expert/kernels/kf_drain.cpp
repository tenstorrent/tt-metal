// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Drains the real flat gate_up_compute.cpp's h (CB 3) on one core (NCRISC). RT: 0 sub-blocks, 1 h tiles a sub-block
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t n = get_arg_val<uint32_t>(0), t = get_arg_val<uint32_t>(1);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(3, t);
        cb_pop_front(3, t);
    }
}
