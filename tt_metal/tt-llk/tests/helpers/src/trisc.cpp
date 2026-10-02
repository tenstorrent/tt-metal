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

#if defined(ARCH_QUASAR)
namespace llk_barrier
{
// barrier.h cannot include profiler.h, so the L1 address is supplied from here.
volatile std::uint32_t* barrier_slots = reinterpret_cast<volatile std::uint32_t*>(llk_profiler::BARRIER_START);
} // namespace llk_barrier
#endif

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

// Reconfig testing applies a config space state written into L1 before the kernel runs.
// The dprint L1 region is reused for this... until we get a better memory map.
#ifndef LLK_DEVICE_PRINT_BUFFER_BASE
static constexpr std::uint32_t RESTORE_PLAN_BASE  = 0x1A000;
static constexpr std::uint32_t RESTORE_PLAN_MAGIC = 0x43464731u; // "CFG1"
static constexpr std::uint32_t RESTORE_SPACE_CONFIG       = 0;
static constexpr std::uint32_t RESTORE_SPACE_THREADCONFIG = 1;
static constexpr std::uint32_t RESTORE_SPACE_ADC_CH1X     = 2;
static constexpr std::uint32_t RESTORE_ENTRY_WORDS        = 6;

// Plan is [magic][N][data], data is N x [space, addr32, v0, v1, v2, mask].
static inline void restore_state()
{
    volatile std::uint32_t* plan = reinterpret_cast<volatile std::uint32_t*>(RESTORE_PLAN_BASE);
    if (plan[0] != RESTORE_PLAN_MAGIC) return;
#if defined(LLK_TRISC_UNPACK)
    constexpr std::uint32_t thread = 0;
#elif defined(LLK_TRISC_MATH)
    constexpr std::uint32_t thread = 1;
#elif defined(LLK_TRISC_PACK)
    constexpr std::uint32_t thread = 2;
#endif
    const std::uint32_t n = plan[1];
    for (std::uint32_t i = 0; i < n; i++)
    {
        const volatile std::uint32_t* e = &plan[2 + RESTORE_ENTRY_WORDS * i];
        const std::uint32_t space       = e[0];
        const std::uint32_t a           = e[1];
        if (space == RESTORE_SPACE_CONFIG)
        {
            const std::uint32_t mask = e[5];
            const std::uint32_t cur  = ckernel::cfg_read(a);
            ckernel::cfg_write(a, (cur & ~mask) | (e[2] & mask));
        }
#if defined(LLK_TRISC_UNPACK) || defined(LLK_TRISC_MATH) || defined(LLK_TRISC_PACK)
        else if (space == RESTORE_SPACE_THREADCONFIG)
        {
            TT_SETC16(a, e[2 + thread] & 0xFFFF);
        }
#endif
#if defined(LLK_TRISC_UNPACK)
        else if (space == RESTORE_SPACE_ADC_CH1X)
        {
            // By unpacker_addr_counter_init, we restore only UNP_A.
            TT_SETADCXY(ckernel::p_setadc::UNP_A, 0, e[2], 0, 0, 0b0100);
        }
#elif defined(LLK_TRISC_PACK)
        else if (space == RESTORE_SPACE_ADC_CH1X)
        {
            TT_SETADCXY(ckernel::p_setadc::PAC, 0, e[3], 0, 0, 0b0100);
        }
#endif
    }
}

#endif

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
    device_setup();
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

#ifndef LLK_DEVICE_PRINT_BUFFER_BASE
        restore_state();
#endif

        run_kernel(temp_args);

        ckernel::fence_compiler();

        ckernel::tensix_sync();
    }

    *mailbox = ckernel::KERNEL_COMPLETE;
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
