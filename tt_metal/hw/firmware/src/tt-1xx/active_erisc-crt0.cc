// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <csetjmp>
#include <cstring>
#include "tt_metal/hw/inc/risc_common.h"
#include "eth_fw_api.h"

void __attribute__((noinline)) Application();

__attribute__((section(".noinit"))) static std::jmp_buf gJumpBuf;

[[gnu::noreturn]] static void return_to_base_fw() {
    // NOLINTNEXTLINE(cert-err52-cpp)
    longjmp(gJumpBuf, 1);
    __builtin_unreachable();
}

// Pointer to exit routine, (so it may be called from a kernel).  USED
// attribute is needed to keep this as an symbol that kernels may
// use. (Because we LTO the firmware, it would otherwise look
// removable.). Only valid when watcher is enabled.
[[gnu::noreturn, gnu::used]] void (*erisc_exit)() = return_to_base_fw;

extern "C" void wzerorange(uint32_t* start, uint32_t* end);

inline void clear_eth_mailbox() {
    volatile uint32_t* const eth_mailbox_base =
    reinterpret_cast<volatile uint32_t*>(MEM_SYSENG_ETH_MAILBOX_ADDR);
    constexpr uint32_t words_per_mailbox = 1 + MEM_SYSENG_ETH_MAILBOX_NUM_ARGS;  // msg + args
    constexpr uint32_t total_words = static_cast<uint32_t>(NUM_ETH_MAILBOX) * words_per_mailbox;

    for (uint32_t i = 0; i < total_words; ++i) {
        eth_mailbox_base[i] = 0;
    }
}

extern "C" [[gnu::section(".start"), gnu::optimize("Os")]] void _start(void) {
    volatile uint32_t* const debug_dump_addr = reinterpret_cast<volatile uint32_t*>(0x36b0);
    for (int i = 0; i < 32; i++) {
        debug_dump_addr[i] = 0xdeadbeef;
    }
    extern uint32_t __ldm_bss_start[];
    extern uint32_t __ldm_bss_end[];
    wzerorange(__ldm_bss_start, __ldm_bss_end);
    erisc_exit = return_to_base_fw;

    // BH-104
    // as of tt-firmware 18.6.0 there is no one else using the mailbox
    // clear the mailbox to clear any potential duplicate PCIe writes before we
    // return to base firmware
    // otherwise base firwmare could launch metal firwmare twice

    // NOLINTNEXTLINE(cert-err52-cpp)
    if (setjmp(gJumpBuf)) {
        // Returned from the longjmp
        clear_eth_mailbox();
        invalidate_l1_cache();
        return;
    } else {
        Application();
        clear_eth_mailbox();
    }

    // Dump values to debug buffer
    // NOLINTNEXTLINE(hicpp-no-assembler)
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
        "nop"
        : /* no output */
        : "r"(debug_dump_addr)
        : "memory");

    invalidate_l1_cache();
}
