// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/device_print.h"

// Tail-call reconstruction: AMBIGUOUS, RESOLVABLE case.
//
// kernel_main -> fork_func -> left -> left_top
// The flow is the following:
// - kernel_main makes an actual call to fork_func()
// - fork_func makes a tail call to left()
// - left makes a tail call to left_top()
//
// At DEVICE_PRINT position:
// - PC resolves to left_top()
// - RA resolves to kernel_main()
//
// The missing frames are RESOLVABLE, because only the left tail call path ends up at left_top()
[[gnu::noinline]] void left_top() {
    std::uintptr_t pc;
    asm volatile("auipc %[pc], 0\n" : [pc] "=r"(pc));
    std::uintptr_t ra = reinterpret_cast<std::uintptr_t>(__builtin_return_address(0));

    DEVICE_PRINT(
        "CALLSTACK_BEGIN\n"
        "{}\n"
        "CALLSTACK_END\n",
        dp_top_callstack_t(pc, ra, 0));
}

[[gnu::noinline]] void left() { [[gnu::musttail]] return left_top(); }

[[gnu::noinline]] void right_top() {
    // this would be an empty function,
    // but that would allow the compiler to optimize away the call,
    // leading to the tail call in fork_func() frame being UNAMBIGUOUS.
    volatile bool prevent_dce = true;
    prevent_dce;
}

[[gnu::noinline]] void right() { [[gnu::musttail]] return right_top(); }

[[gnu::noinline]] void fork_func() {
    volatile bool take_left = true;
    if (take_left) {
        [[gnu::musttail]] return left();
    } else {
        [[gnu::musttail]] return right();
    }
}

void kernel_main() {
    volatile int force_real_call = 0;
    fork_func();
    force_real_call;  // prevent tail-call of fork_func() from kernel_main()
}
