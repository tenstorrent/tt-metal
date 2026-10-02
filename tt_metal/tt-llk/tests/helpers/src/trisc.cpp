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
static constexpr std::uint32_t LLK_RESTORE_PLAN_BASE  = 0x1A000;
static constexpr std::uint32_t LLK_RESTORE_PLAN_MAGIC = 0x52535431u; // "RST1"

// Apply the passed config plan that starts at BASE.
// The plan is stored as [magic][N][data], where data is N x [addr32, value, port, mask].
static inline void apply_plan_at(std::uint32_t base, std::uint32_t magic)
{
    volatile std::uint32_t* plan = reinterpret_cast<volatile std::uint32_t*>(base);
    if (plan[0] != magic)
    {
        return;
    }
    const std::uint32_t n = plan[1];
    for (std::uint32_t i = 0; i < n; i++)
    {
        const std::uint32_t a    = plan[2 + 4 * i + 0];
        const std::uint32_t v    = plan[2 + 4 * i + 1];
        const std::uint32_t port = plan[2 + 4 * i + 2];
        const std::uint32_t mask = plan[2 + 4 * i + 3];
        if (port == 1)
        {
            TT_SETC16(a, v & mask & 0xFFFF);
        }
        else
        {
            const std::uint32_t cur = ckernel::cfg_read(a);
            ckernel::cfg_write(a, (cur & ~mask) | (v & mask));
        }
    }
}

// Plan format is [magic][N][has_ch1x][data], where data is N x [addr32, v_t0, v_t1, v_t2],
// and, if has_ch1x is set, [adc_ch1x_unpacker][adc_ch1x_packer] at the very end.
static constexpr std::uint32_t LLK_RESTORE_ADDRMOD_BASE  = 0x1C000;
static constexpr std::uint32_t LLK_RESTORE_ADDRMOD_MAGIC = 0x41525431u; // 'ART1'

static inline void apply_addrmod_restore()
{
    volatile std::uint32_t* plan = reinterpret_cast<volatile std::uint32_t*>(LLK_RESTORE_ADDRMOD_BASE);
    if (plan[0] != LLK_RESTORE_ADDRMOD_MAGIC)
    {
        return;
    }
#if defined(LLK_TRISC_UNPACK)
    constexpr std::uint32_t my_thread = 0;
#elif defined(LLK_TRISC_MATH)
    constexpr std::uint32_t my_thread = 1;
#elif defined(LLK_TRISC_PACK)
    constexpr std::uint32_t my_thread = 2;
#else
    return;
#endif
    const std::uint32_t n        = plan[1];
    const std::uint32_t has_ch1x = plan[2];
    for (std::uint32_t i = 0; i < n; i++)
    {
        const std::uint32_t a = plan[3 + 4 * i + 0];
        const std::uint32_t v = plan[3 + 4 * i + 1 + my_thread];
        TT_SETC16(a, v & 0xFFFF);
    }
    if (!has_ch1x)
    {
        return;
    }
#if defined(LLK_TRISC_UNPACK)
    // By unpacker_addr_counter_init, we restore only UNP_A.
    TT_SETADCXY(ckernel::p_setadc::UNP_A, 0, plan[3 + 4 * n + 0], 0, 0, 0b0100);
#elif defined(LLK_TRISC_PACK)
    TT_SETADCXY(ckernel::p_setadc::PAC, 0, plan[3 + 4 * n + 1], 0, 0, 0b0100);
#endif
}

static inline void apply_restore_plan()
{
    apply_plan_at(LLK_RESTORE_PLAN_BASE, LLK_RESTORE_PLAN_MAGIC);
    apply_addrmod_restore();
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
        apply_restore_plan();
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
