// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Test kernel (tests/dfl_long_predecessor.py): keeps a core busy writing a pattern over its CB 0, `passes` times.
// Run on one chip right before a resident step, it stands in for a chip still running another program (the drafter's
// context fill) while its peers start the step and push their first hub round into this chip's hub slots, which sit
// at the same L1 addresses (the slots are the hub core's CB 0, mlp_hub.cpp).
// Compile-time args: 0 bytes (CB 0's size). Runtime args: 0 passes, 1 pattern word.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t bytes = get_compile_time_arg_val(0);
    const uint32_t passes = get_arg_val<uint32_t>(0);
    const uint32_t pattern = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
    for (uint32_t k = 0; k < passes; k++) {
        for (uint32_t i = 0; i < bytes / 4; i++) {
            p[i] = pattern;
        }
    }
}
