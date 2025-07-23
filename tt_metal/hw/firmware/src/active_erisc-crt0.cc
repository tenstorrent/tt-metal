// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <csetjmp>
#include <cstdlib>

void __attribute__((noinline)) Application();

static std::jmp_buf jump_buf;

// This function is noreturn, it will not return to the caller back to _start
[[gnu::noreturn]] static void return_to_base_fw() {
    // NOLINTNEXTLINE(cert-err52-cpp)
    longjmp(jump_buf, 1);
    __builtin_unreachable();
}

// Pointer to exit routine, (so it may be called from a kernel).  USED
// attribute is needed to keep this as an symbol that kernels may
// use. (Because we LTO the firmware, it would otherwise look
// removable.)
[[gnu::noreturn, gnu::used]] void (*erisc_exit)() = return_to_base_fw;

extern "C" void wzerorange(uint32_t* start, uint32_t* end);

extern "C" [[gnu::section(".start")]] void _start(void) {
    // Clear bss before using any globals
    extern uint32_t __ldm_bss_start[];
    extern uint32_t __ldm_bss_end[];
    wzerorange(__ldm_bss_start, __ldm_bss_end);

    // NOLINTNEXTLINE(cert-err52-cpp)
    int return_value = setjmp(jump_buf);
    if (return_value == 0) {
        Application();
    }
}
