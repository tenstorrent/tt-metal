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
barrier_ptr_t barrier_ptr          = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
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

#if defined(LLK_PROFILER)
    llk_profiler::reset();
#endif

    // Wormhole L1_TO_L1 dithering: the host asks for several repetitions of the kernel in one launch, each at
    // its own start offset, so one launch samples several packer phases at the cost of one.
    const volatile std::uint32_t* dither = reinterpret_cast<volatile std::uint32_t*>(llk_perf::PERF_START_OFFSET_ADDR);
    const std::uint32_t repetitions      = dither[0] != 0 ? dither[0] : 1;
    const std::uint32_t pack_grid        = dither[1];
    const std::uint32_t other_grid       = dither[2];
    const std::uint32_t pack_count       = (pack_grid >> 16) != 0 ? (pack_grid >> 16) : 1;
    const std::uint32_t other_count      = (other_grid >> 16) != 0 ? (other_grid >> 16) : 1;

    for (std::uint32_t repetition = 0; repetition < repetitions; ++repetition)
    {
        llk_perf::detail::start_offset_word = ((repetition % pack_count) * (pack_grid & 0xFFFFu)) |
                                              ((((repetition / pack_count) % other_count) * (other_grid & 0xFFFFu)) << 16);

        std::fill(ckernel::regfile, ckernel::regfile + 64, 0);

#ifndef ARCH_QUASAR
        ckernel::reset_cfg_state_id();
        ckernel::reset_dest_offset_id();
#endif

#if defined(LLK_PROFILER)
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
    }

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