// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/device_print.h"

void kernel_main() {
    // In case _start -> run_kernel -> kernel_main gets inlined, RA will be pointing into firmware.
    // The callstack should only print kernel_main and stop because it's a terminal frame.
    std::uintptr_t pc;
    asm volatile("auipc %[pc], 0\n" : [pc] "=r"(pc));
    std::uintptr_t ra = reinterpret_cast<std::uintptr_t>(__builtin_return_address(0));

    DEVICE_PRINT(
        "CALLSTACK_BEGIN\n"
        "{}\n"
        "CALLSTACK_END\n",
        dp_top_callstack_t(pc, ra, 0));
}
