// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/debug/device_print.h"

// Tail-call reconstruction: AMBIGUOUS, UNRESOLVABLE case.
//
// kernel_main -> fork_func -> {left, right} -> leaf
// The flow is the following:
// - kernel_main makes an actual call to fork_func()
// - fork_func makes a tail call to either left() or right()
// - both left() and right() make a tail call to the same leaf()
//
// At DEVICE_PRINT position:
// - PC resolves to leaf()
// - RA resolves to kernel_main()
//
// The missing frames are UNRESOLVABLE
// because the left/right path can't be inferred from leaf
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

[[gnu::noinline]] void left() {
    // we prevent identical code folding between left() and right()
    // to ensure that the calls in fork_func() are ambiguous,
    // and don't optimize into a single tail call.
    volatile int prevent_icf = 0;
    prevent_icf;
    [[gnu::musttail]] return leaf();
}

[[gnu::noinline]] void right() {
    volatile int prevent_icf = 1;
    prevent_icf;
    [[gnu::musttail]] return leaf();
}

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
