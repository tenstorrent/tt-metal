// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <csetjmp>

int __attribute__((noinline)) main();

static std::jmp_buf gJumpBuf;

static void return_to_base_fw();

// Pointer to exit routine, (so it may be called from a kernel).  USED
// attribute is needed to keep this as an symbol that kernels may
// use. (Because we LTO the firmware, it would otherwise look
// removable.). Only valid when watcher is enabled.
[[gnu::noreturn, gnu::used]] void (*erisc_exit)() = return_to_base_fw;

extern "C" void wzerorange(uint32_t* start, uint32_t* end);

extern "C" [[gnu::section(".start"), gnu::optimize("Os")]] void _start(void) {
    extern uint32_t __ldm_bss_start[];
    extern uint32_t __ldm_bss_end[];
    wzerorange(__ldm_bss_start, __ldm_bss_end);

    // NOLINTNEXTLINE(cert-err52-cpp)
    if (setjmp(gJumpBuf)) {
        // Returned from the longjmp
        // Do not run main() again
    } else {
        main();
    }
    __asm__ volatile("fence");
}

static void return_to_base_fw() {
    // NOLINTNEXTLINE(cert-err52-cpp)
    longjmp(gJumpBuf, 1);
    __builtin_unreachable();
}
