// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RC experiment switches: which half of the L1 port-3 fix is on
#define LLK_PORT3_AT_END   1
#define LLK_PORT3_AT_START 1
#include <algorithm>
#include <cstdint>

#include "ckernel.h"
#ifndef ARCH_QUASAR
#include "ckernel_globals.h" // Only for WH/BH
#include "llk_assert.h"
// Necessary for ckernel variables
#include "ckernel_helper.h" // Only for WH/BH
#endif
#include "boot.h"
#include "counters.h"
#include "profiler.h"

#ifdef LLK_PROFILER

namespace llk_profiler
{
buffer_ptr_t buffer                = reinterpret_cast<buffer_ptr_t>(BUFFERS_START);
epoch_ptr_t epoch_ptr              = reinterpret_cast<epoch_ptr_t>(EPOCH_ADDR);
std::uint32_t write_idx            = 0;
std::uint32_t reserved_words_count = 0;

} // namespace llk_profiler

#endif

// Mailbox addresses
#ifdef COVERAGE
extern "C"
{
    extern void gcov_dump(void);
}
constexpr std::uint32_t mailboxes_start = 0x6DFB8;
#else
constexpr std::uint32_t mailboxes_start = 0x1FFB8;
#endif

#if defined(LLK_TRISC_UNPACK)
constexpr std::uint32_t mailbox_offset = 0;
#elif defined(LLK_TRISC_MATH)
constexpr std::uint32_t mailbox_offset = sizeof(std::uint32_t);
#elif defined(LLK_TRISC_PACK)
constexpr std::uint32_t mailbox_offset = 2 * sizeof(std::uint32_t);
#elif defined(LLK_TRISC_ISOLATE_SFPU)
constexpr std::uint32_t mailbox_offset = 3 * sizeof(std::uint32_t);
#else
#error "No TRISC define set"
#endif

void copy_runtimes_from_L1(struct RuntimeParams* temp_args)
{
    extern const volatile struct RuntimeParams __runtime_args_start[];
    ckernel::memcpy_blocking(temp_args, __runtime_args_start, sizeof(struct RuntimeParams));
}

int main(void)
{
    mailbox_t mailbox = reinterpret_cast<volatile std::uint32_t*>(mailboxes_start + mailbox_offset);
#if defined(LLK_TRISC_UNPACK) && defined(LLK_BOOT_MODE_TRISC)
    mailbox_t mailbox_base = reinterpret_cast<volatile std::uint32_t*>(mailboxes_start);
    *(mailbox_base)        = ckernel::RESET_VAL;
    *(mailbox_base + 1)    = ckernel::RESET_VAL;
    *(mailbox_base + 2)    = ckernel::RESET_VAL;
#ifdef ARCH_QUASAR
    *(mailbox_base + 3) = ckernel::RESET_VAL;
#endif
#if defined(ARCH_WORMHOLE) || defined(ARCH_BLACKHOLE)
    for (std::uint32_t slot = 0; slot < 3; ++slot)
    {
        host_signal::write(slot, ckernel::RESET_VAL);
    }
#endif
    device_setup();
#if defined(ARCH_QUASAR)
    // No BRISC on Quasar: unpack configures and arms the counters while the other TRISCs are still held in reset.
    llk_perf::configure_and_arm();
#endif
    clear_trisc_soft_reset(); // Release the rest of the triscs
#endif

    struct RuntimeParams temp_args;
    copy_runtimes_from_L1(&temp_args);

    std::fill(ckernel::regfile, ckernel::regfile + 64, 0);

#ifndef ARCH_QUASAR
    ckernel::reset_cfg_state_id();
    ckernel::reset_dest_offset_id();
#endif

#if defined(LLK_PROFILER)
    llk_profiler::reset();
    llk_profiler::sync_threads();
#endif

#if defined(ARCH_WORMHOLE) && defined(LLK_PORT3_AT_START)
    // Start-of-kernel half of the port-3 fix (covers the first kernel after anything else ran on the core): all
    // three TRISCs meet; while math and unpack spin on the release semaphore (PC buffer, not L1) the pack TRISC
    // waits for them to settle and makes one L1 load, so it is the last port-3 requester before the kernel body.
    llk_barrier::rendezvous(
        llk_barrier::is_action_thread(),
        []
        {
            constexpr std::uint64_t PORT3_START_SETTLE_CYCLES = 512;
            const std::uint64_t settle_start                  = ckernel::read_wall_clock();
            while (ckernel::read_wall_clock() - settle_start < PORT3_START_SETTLE_CYCLES)
            {
            }
            (void)ckernel::load_blocking(reinterpret_cast<volatile std::uint32_t*>(mailboxes_start));
        });
#endif

    {
        ZONE_SCOPED("KERNEL")

        ckernel::fence_compiler();

        run_kernel(temp_args);

        ckernel::fence_compiler();

        ckernel::tensix_sync();
    }

    llk_perf::read_last_zone();

#if defined(ARCH_WORMHOLE) && defined(LLK_TRISC_PACK) && defined(LLK_PORT3_AT_END)
    // Wormhole L1 port 3 is a round-robin arbiter over {0: TDMA (packer L1 writes, mover), 1: math TRISC,
    // 2: pack TRISC}. Its last-grant pointer survives TRISC soft reset and is cleared only by a tile reset, so the
    // L1 access rhythm of the next kernel (and its perf numbers) depended on which requester this kernel used last.
    // Make the pack TRISC the last requester every time: wait until the math TRISC has done its last L1 store
    // (its completion mailbox), let it settle into its end loop (it then runs from the icache), then load one L1
    // word. The packer is idle here (tensix_sync), and nothing after this on port 3 comes from math or TDMA.
    {
        mailbox_t math_mailbox = reinterpret_cast<volatile std::uint32_t*>(mailboxes_start + sizeof(std::uint32_t));
        while (ckernel::load_blocking(math_mailbox) != ckernel::KERNEL_COMPLETE)
        {
        }
        constexpr std::uint64_t PORT3_SETTLE_CYCLES = 4096;
        const std::uint64_t settle_start            = ckernel::read_wall_clock();
        while (ckernel::read_wall_clock() - settle_start < PORT3_SETTLE_CYCLES)
        {
        }
        (void)ckernel::load_blocking(math_mailbox);
    }
#endif

    *mailbox = ckernel::KERNEL_COMPLETE;
#if defined(ARCH_WORMHOLE) || defined(ARCH_BLACKHOLE)
    host_signal::write(mailbox_offset / sizeof(std::uint32_t), ckernel::KERNEL_COMPLETE);
#endif
}

extern "C" __attribute__((section(".init"), naked, noreturn, no_profile_instrument_function)) std::uint32_t _start()
{
    do_crt0();

    main();

#ifdef COVERAGE
    gcov_dump();
#endif

    for (;;)
    {
    } // Loop forever
}
