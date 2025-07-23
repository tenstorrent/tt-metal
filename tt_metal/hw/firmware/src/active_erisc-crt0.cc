// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <csetjmp>
#include <cstring>

void __attribute__((noinline)) Application();

static std::jmp_buf gJumpBuf;

// This function is noreturn, it will not return to the caller back to _start
[[gnu::noreturn]] static void return_to_base_fw() {
    // NOLINTNEXTLINE(cert-err52-cpp)
    longjmp(gJumpBuf, 1);
    __builtin_unreachable();
}

// Pointer to exit routine, (so it may be called from a kernel).  USED
// attribute is needed to keep this as an symbol that kernels may
// use. (Because we LTO the firmware, it would otherwise look
// removable.)
[[gnu::noreturn, gnu::used]] void (*erisc_exit)() = return_to_base_fw;

extern "C" void wzerorange(uint32_t* start, uint32_t* end);

extern "C" [[gnu::section(".start")]] void _start(void) {
    // The C++ calling convention saves ra to the stack, but we can't rely on that
    // because this is the entry point. We must save it manually before any C++
    // function calls.
    std::jmp_buf jump_buf;
    // NOLINTNEXTLINE(cert-err52-cpp)
    int return_value = setjmp(jump_buf);
    if (return_value == 0) {
        volatile uint32_t* const reg_dump_addr = reinterpret_cast<volatile uint32_t*>(0x36b0);
        for (int i = 0; i < 13; i++) {
            reg_dump_addr[i] = 0;
        }
        // Clear bss before using any globals
        extern uint32_t __ldm_bss_start[];
        extern uint32_t __ldm_bss_end[];
        wzerorange(__ldm_bss_start, __ldm_bss_end);
        memcpy(&gJumpBuf, &jump_buf, sizeof(jump_buf));
        Application();
    } else {
        // Dump the RISCV registers to memory at address for debugging purposes
        volatile uint32_t* const reg_dump_addr = reinterpret_cast<volatile uint32_t*>(0x36b0);
        __asm__ volatile(
            "sw ra, 0 * 4(%0)\n\t"
            "sw s0, 1 * 4(%0)\n\t"
            "sw s1, 2 * 4(%0)\n\t"
            "sw s2, 3 * 4(%0)\n\t"
            "sw s3, 4 * 4(%0)\n\t"
            "sw s4, 5 * 4(%0)\n\t"
            "sw s5, 6 * 4(%0)\n\t"
            "sw s6, 7 * 4(%0)\n\t"
            "sw s7, 8 * 4(%0)\n\t"
            "sw s8, 9 * 4(%0)\n\t"
            "sw s9, 10 * 4(%0)\n\t"
            "sw s10, 11 * 4(%0)\n\t"
            "sw s11, 12 * 4(%0)\n\t"
            : /* no output */
            : "r"(reg_dump_addr)
            : "memory");
    }
}
