// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/device_print.h"

// Test that after unwinding PC and RA, we terminate the callstack after finding terminal.
[[gnu::always_inline]] inline void pc3() {
    std::uintptr_t pc;
    asm volatile("auipc %[pc], 0\n" : [pc] "=r"(pc));
    std::uintptr_t ra = reinterpret_cast<std::uintptr_t>(__builtin_return_address(0));

    DEVICE_PRINT(
        "CALLSTACK_BEGIN\n"
        "{}\n"
        "CALLSTACK_END\n",
        dp_top_callstack_t(pc, ra, 0));
}

[[gnu::always_inline]] inline void pc2() { pc3(); }

[[gnu::noinline]] inline void pc1() { pc2(); }

[[gnu::always_inline]] inline void ra3() { pc1(); }

[[gnu::always_inline]] inline void ra2() { ra3(); }

[[gnu::always_inline]] inline void ra1() { ra2(); }

void kernel_main() { ra1(); }
