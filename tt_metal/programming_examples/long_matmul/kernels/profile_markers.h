// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "api/debug/kernel_profiler.hpp"

// Kernel-level timing for the tt-ember parser (tools/tt-ember/parser.py, added in #58134; its
// RE_KERNEL_ZONE and RE_KERNEL_CYCLES match these names).
//
// With the device profiler on (TT_METAL_DEVICE_PROFILER=1, which makes the JIT build define
// PROFILE_KERNEL), each kernel records <KERNEL>_KERNEL_START and <KERNEL>_KERNEL_END once per
// invocation, and just before the end the total cycles it spent blocked on its circular buffers
// (<KERNEL>_WAIT_CYCLES) and doing its real work (<KERNEL>_TRANSFER_CYCLES or
// COMPUTE_COMPUTE_CYCLES). Everything else is "overhead", which parser.py derives on the host.
//
// Without the profiler every helper here compiles to nothing, so the instruction stream of the
// measured workload is unchanged.
namespace long_matmul_profile {

#if defined(PROFILE_KERNEL)
constexpr bool kEnabled = true;
#else
constexpr bool kEnabled = false;
#endif

// Only the low 32 bits of the wall clock are read. Taking both halves can tear (the high word was
// seen to come back one ahead, adding 2^32 cycles to a span), and every span measured here is far
// shorter than 2^32 cycles (about 4 s at 1 GHz), so a 32-bit unsigned difference is exact.
inline uint32_t now() {
    if constexpr (kEnabled) {
        return get_timestamp_32b();
    } else {
        return 0;
    }
}

inline void add_since(uint64_t& total, uint32_t start) {
    if constexpr (kEnabled) {
        total += static_cast<uint32_t>(get_timestamp_32b() - start);
    }
}

}  // namespace long_matmul_profile
