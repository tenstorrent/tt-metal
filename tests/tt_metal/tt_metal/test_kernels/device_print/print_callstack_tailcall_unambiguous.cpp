// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/device_print.h"

// Tail-call reconstruction: UNAMBIGUOUS case.
//
// kernel_main -> top -> mid -> leaf
// The flow is the following:
// - kernel_main makes an actual call to top()
// - top makes a tail call to mid()
// - mid makes a tail call to leaf()
//
// At DEVICE_PRINT position:
// - PC resolves to leaf()
// - RA resolves to kernel_main()
//
// The missing frames will be reconstructed from the DWARF call graph.
[[gnu::noinline]] void leaf() {
    std::uintptr_t pc;
    asm volatile("auipc %[pc], 0\n" : [pc] "=r"(pc));
    std::uintptr_t ra = reinterpret_cast<std::uintptr_t>(__builtin_return_address(0));

    DEVICE_PRINT(
        "CALLSTACK_BEGIN\n"
        "{}\n"
        "CALLSTACK_END\n",
        dp_top_callstack_t(pc, ra, 0));
}

[[gnu::noinline]] void mid() { [[gnu::musttail]] return leaf(); }

[[gnu::noinline]] void top() { [[gnu::musttail]] return mid(); }

void kernel_main() {
    volatile int force_real_call = 0;
    top();
    force_real_call;  // prevent tail-call of top() from kernel_main()
}
