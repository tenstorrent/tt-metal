// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/device_print.h"

volatile int dummy;

// skip_frames = 4 spans both callstacks.
// Should skip pc2, pc1, ra5, ra4
// Should print ra3, ra2, ra1, kernel_main
// This exercises a skip count that crosses the PC/RA boundary.
[[gnu::always_inline]] inline void pc2() {
    std::uintptr_t pc;
    asm volatile("auipc %[pc], 0\n" : [pc] "=r"(pc));
    std::uintptr_t ra = reinterpret_cast<std::uintptr_t>(__builtin_return_address(0));

    DEVICE_PRINT(
        "CALLSTACK_BEGIN\n"
        "{}\n"
        "CALLSTACK_END\n",
        dp_top_callstack_t(pc, ra, 4));
}

[[gnu::noinline]] void pc1() { pc2(); }

[[gnu::always_inline]] inline void ra5() {
    pc1();
    // prevent TCO from collapsing the pc1 call frame
    dummy = 0;
}

[[gnu::always_inline]] inline void ra4() { ra5(); }

[[gnu::always_inline]] inline void ra3() { ra4(); }

[[gnu::always_inline]] inline void ra2() { ra3(); }

[[gnu::always_inline]] inline void ra1() { ra2(); }

void kernel_main() { ra1(); }
