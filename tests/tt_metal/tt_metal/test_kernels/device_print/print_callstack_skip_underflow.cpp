// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/device_print.h"

volatile int dummy;

// Skip count that exceeds the total number of frames should leave a continuation sentinel.
[[gnu::always_inline]] inline void inner() {
    std::uintptr_t pc;
    asm volatile("auipc %[pc], 0\n" : [pc] "=r"(pc));
    std::uintptr_t ra = reinterpret_cast<std::uintptr_t>(__builtin_return_address(0));

    DEVICE_PRINT(
        "CALLSTACK_BEGIN\n"
        "{}\n"
        "CALLSTACK_END\n",
        dp_top_callstack_t(pc, ra, 3));
}

[[gnu::noinline]] void middle() {
    inner();
    // prevent TCO from collapsing the inner call frame
    dummy = 0;
}

void kernel_main() {
    middle();
    // prevent TCO from collapsing the middle call frame
    dummy = 0;
}
