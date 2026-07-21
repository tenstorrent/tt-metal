// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/device_print.h"

// Test case where the stack can be terminated jusst by unwindind inline frames from PC.
// We poison the RA to an non-existant value to ensure we only rely on PC in this case.
[[gnu::always_inline]] inline void pc3() {
    std::uintptr_t pc;
    asm volatile("auipc %[pc], 0\n" : [pc] "=r"(pc));

    DEVICE_PRINT(
        "CALLSTACK_BEGIN\n"
        "{}\n"
        "CALLSTACK_END\n",
        dp_top_callstack_t(pc, 0xDEADBEEF, 0));
}

[[gnu::always_inline]] inline void pc2() { pc3(); }

[[gnu::always_inline]] inline void pc1() { pc2(); }

void kernel_main() { pc1(); }
