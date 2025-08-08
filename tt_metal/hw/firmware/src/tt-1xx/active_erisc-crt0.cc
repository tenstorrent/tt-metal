// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <csetjmp>
#include <cstring>
#include "tt_metal/hw/inc/risc_common.h"
#include "eth_fw_api.h"

void __attribute__((noinline)) Application();

static std::jmp_buf gJumpBuf;

static void return_to_base_fw();

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
    extern uint32_t __ldm_bss_start[];
    extern uint32_t __ldm_bss_end[];
    wzerorange(__ldm_bss_start, __ldm_bss_end);

    // BH-104
    // as of tt-firmware 18.6.0 there is no one else using the mailbox
    // clear the mailbox to clear any potential duplicate PCIe writes before we
    // return to base firmware
    // otherwise base firwmare could launch metal firwmare twice

    // NOLINTNEXTLINE(cert-err52-cpp)
    if (setjmp(gJumpBuf)) {
        // Returned from the longjmp
        // Do not run Application() again
        __asm__ volatile("nop; nop; nop; nop;");
    } else {
        Application();
    }

    clear_eth_mailbox();
    invalidate_l1_cache();
}

static void return_to_base_fw() {
    // NOLINTNEXTLINE(cert-err52-cpp)
    longjmp(gJumpBuf, 1);
    __builtin_unreachable();
}
