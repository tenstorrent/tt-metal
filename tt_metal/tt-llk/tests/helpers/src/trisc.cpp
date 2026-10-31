// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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

    {
        ZONE_SCOPED("KERNEL")

        ckernel::fence_compiler();

        run_kernel(temp_args);

        ckernel::fence_compiler();

        ckernel::tensix_sync();
    }

    llk_perf::read_last_zone();

    *mailbox = ckernel::KERNEL_COMPLETE;
#if defined(ARCH_WORMHOLE) || defined(ARCH_BLACKHOLE)
    host_signal::write(mailbox_offset / sizeof(std::uint32_t), ckernel::KERNEL_COMPLETE);
#endif
}

#if defined(ARCH_WORMHOLE) && !defined(COVERAGE)
#define LLK_PORT3_TAIL 1
#ifndef LLK_PORT3_LAST
#define LLK_PORT3_LAST 1 // L1 port-3 requester index that makes the last access of every kernel: 1 math, 2 pack TRISC
#endif

// Wormhole L1 port 3 is a round-robin arbiter over {0: TDMA (packer L1 writes, mover), 1: math TRISC, 2: pack TRISC}.
// Its last-grant pointer survives the TRISC soft reset (only a tile reset clears it), and the next kernel's L1 access
// rhythm, so its perf numbers, depended on which TRISC this kernel used port 3 last. Here one TRISC (LLK_PORT3_LAST)
// always makes the last access: after main it waits for the other TRISC's completion mailbox (its last L1 store), lets
// it settle into this end loop (which then runs from the icache), and loads one L1 word. The packer is idle by then
// (tensix_sync). This code sits after all other code (section port3_text, placed last in .text by sections.ld) and
// _start reaches it with a jump of the same size as its old end loop, so main and the kernel keep their addresses.
extern "C" [[noreturn]] __attribute__((noinline, used, section("port3_text"), no_profile_instrument_function)) void port3_tail()
{
    constexpr std::uint32_t me = mailbox_offset / sizeof(std::uint32_t); // 0 unpack, 1 math, 2 pack
    if constexpr (me == LLK_PORT3_LAST)
    {
        mailbox_t peer_mailbox = reinterpret_cast<volatile std::uint32_t*>(mailboxes_start) + (3 - me);
        while (ckernel::load_blocking(peer_mailbox) != ckernel::KERNEL_COMPLETE)
        {
        }
        constexpr std::uint64_t PORT3_SETTLE_CYCLES = 4096;
        const std::uint64_t settle_start            = ckernel::read_wall_clock();
        while (ckernel::read_wall_clock() - settle_start < PORT3_SETTLE_CYCLES)
        {
        }
        (void)ckernel::load_blocking(peer_mailbox);
    }
    for (;;)
    {
    }
}
#endif

extern "C" __attribute__((section(".init"), naked, noreturn, no_profile_instrument_function)) std::uint32_t _start()
{
    do_crt0();

    main();

#ifdef COVERAGE
    gcov_dump();
#endif

#if defined(LLK_PORT3_TAIL)
    port3_tail();
#else
    for (;;)
    {
    } // Loop forever
#endif
}
