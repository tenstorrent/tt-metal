// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel for KERNEL_PROFILER structural zone-id budgeting. It emits many distinctly-named
// device profiler zones in a single translation unit:
//   - by default: 120 zones, comfortably under the per-TU budget (KERNEL_PROFILER_LOCAL_BITS = 7 =>
//     128 zone ids per TU, minus the 3 internal PROFILER-* zones);
//   - with -DZONE_OVER_BUDGET: 130 zones, which pushes the per-TU local index past 127 and must trip
//     the static_assert in TT_ZONE_META (kernel_profiler.hpp) so the kernel fails to compile.
//
// This is a data-movement kernel, so the profiler macros are auto-injected by the JIT build and no
// explicit profiler include is needed.

#include <cstdint>

// One profiler zone in its own block scope, with a unique name via adjacent string-literal
// concatenation. Distinct call sites get distinct __COUNTER__ values -> distinct structural ids.
#define ZONE(nm)               \
    {                          \
        DeviceZoneScopedN(nm); \
        asm volatile("nop");   \
    }
#define ZONE10(p) \
    ZONE(p "0")   \
    ZONE(p "1") ZONE(p "2") ZONE(p "3") ZONE(p "4") ZONE(p "5") ZONE(p "6") ZONE(p "7") ZONE(p "8") ZONE(p "9")

void kernel_main() {
    // 120 distinctly-named zones (12 groups of 10: a0..l9).
    ZONE10("a")
    ZONE10("b")
    ZONE10("c")
    ZONE10("d")
    ZONE10("e")
    ZONE10("f")
    ZONE10("g")
    ZONE10("h")
    ZONE10("i")
    ZONE10("j")
    ZONE10("k")
    ZONE10("l")

#if defined(ZONE_OVER_BUDGET)
    // 10 more -> 130 zones in this TU -> per-TU local index exceeds 127 -> static_assert must fire.
    ZONE10("z")
#endif
}
