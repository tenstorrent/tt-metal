// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <csetjmp>
#include <dev_mem_map.h>

int __attribute__((noinline)) main();

static std::jmp_buf gJumpBuf;

static void return_to_base_fw();

// Pointer to exit routine, (so it may be called from a kernel).  USED
// attribute is needed to keep this as an symbol that kernels may
// use. (Because we LTO the firmware, it would otherwise look
// removable.). Only valid when watcher is enabled.
[[gnu::noreturn, gnu::used]] void (*erisc_exit)() = return_to_base_fw;

extern "C" void wzerorange(uint32_t* start, uint32_t* end);
#define STR(x) #x
#define XSTR(s) STR(s)
#define XSTR2(s) XSTR(s)
#define MEM_NCRISC_HALT_STACK_MAILBOX_ADDRESS (MEM_AERISC_MAILBOX_BASE + 4)
extern "C" __attribute__((naked, used)) void resume_from_reset() {
    __asm__ volatile(
	"lw  sp, " XSTR2(MEM_NCRISC_HALT_STACK_MAILBOX_ADDRESS) "( zero )\n"
	"lw  ra, 0 * 4( sp )\n"
	"lw  s0, 1 * 4( sp )\n"
	"lw  s1, 2 * 4( sp )\n"
	"lw  s2, 3 * 4( sp )\n"
	"lw  s3, 4 * 4( sp )\n"
	"lw  s4, 5 * 4( sp )\n"
	"lw  s5, 6 * 4( sp )\n"
	"lw  s6, 7 * 4( sp )\n"
	"lw  s7, 8 * 4( sp )\n"
	"lw  s8, 9 * 4( sp )\n"
	"lw  s9, 10 * 4( sp )\n"
	"lw  s10, 11 * 4( sp )\n"
	"lw  s11, 12 * 4( sp )\n"
	"lw  a0, 13 * 4( sp )\n"
	"addi sp, sp, (16 * 4)\n"
	"j    _real_start\n"
    );
}

extern "C" __attribute__((naked)) [[gnu::section(".start")]] void _start(void) {
    __asm__ volatile(
        "addi sp, sp, -(16 * 4)\n"
	"sw ra, 0 * 4( sp )\n"  
	"sw s0, 1 * 4( sp )\n"
	"sw s1, 2 * 4( sp )\n"
	"sw s2, 3 * 4( sp )\n"
	"sw s3, 4 * 4( sp )\n"
	"sw s4, 5 * 4( sp )\n"
	"sw s5, 6 * 4( sp )\n"
	"sw s6, 7 * 4( sp )\n"
	"sw s7, 8 * 4( sp )\n"
	"sw s8, 9 * 4( sp )\n"
	"sw s9, 10 * 4( sp )\n"
	"sw s10, 11 * 4( sp )\n"
	"sw s11, 12 * 4( sp )\n"
	"sw a1,  13 * 4( sp )\n"
	"sw sp, " XSTR2(MEM_NCRISC_HALT_STACK_MAILBOX_ADDRESS) "( zero )\n"
    "j resume_from_reset\n"
    );
}

extern "C" __attribute__((used))  void _real_start(void) {
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
