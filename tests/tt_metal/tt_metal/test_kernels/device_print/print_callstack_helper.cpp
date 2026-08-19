// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/device_print.h"

void kernel_main() {
    DEVICE_PRINT(
        "CALLSTACK_BEGIN\n"
        "{}\n"
        "CALLSTACK_END\n",
        dp_top_callstack_t::current(0));
}
