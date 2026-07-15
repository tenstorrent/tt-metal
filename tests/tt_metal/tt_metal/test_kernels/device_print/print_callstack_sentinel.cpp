// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/device_print.h"

// Both PC and RA are the invalid-address sentinel, and a non-zero skip count is requested. Neither
// address resolves, so the host should bail out to the single "..." sentinel frame regardless of
// skip_frames.
void kernel_main() {
    DEVICE_PRINT(
        "CALLSTACK_BEGIN\n"
        "{}\n"
        "CALLSTACK_END\n",
        dp_top_callstack_t(UINTPTR_MAX, UINTPTR_MAX, 4));
}
