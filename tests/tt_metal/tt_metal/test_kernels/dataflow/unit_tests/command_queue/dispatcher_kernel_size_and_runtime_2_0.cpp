// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 variant of dispatcher_kernel_size_and_runtime.cpp. What the dispatch stress tests need
// from this kernel is that it occupies the core for a while and that its binary is a given size, so
// this keeps the spin loop and the .text padding. Gen2 has no c_tensix_core wall clock, so it spins
// on the RISC-V cycle counter instead.
//
// The Gen1 kernel also verifies runtime args, semaphore values and circular buffer page sizes. Those
// have no equivalent here yet, so the host must only use this kernel for the argument-free shape;
// see the guard in random_program_fixture.hpp.

#include <cstdint>

#if KERNEL_SIZE_BYTES > 16
constexpr uint32_t empty_kernel_bytes = 16;
[[gnu::section(".text"), gnu::used]]
static uint8_t lorem_ipsum[KERNEL_SIZE_BYTES - empty_kernel_bytes];
#endif

static inline uint32_t rdcycle() {
    uint32_t cycles;
    asm volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
}

void kernel_main() {
    // Unsigned subtraction so the comparison stays correct across a counter wrap.
    const uint32_t start_time = rdcycle();
    while (rdcycle() - start_time < KERNEL_RUNTIME_MICROSECONDS);
}
