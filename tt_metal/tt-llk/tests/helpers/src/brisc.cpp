// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <type_traits>

// Ick! We have to tell ckernel_ops.h that we're using TTI_ macros.
// This is just wrong and should be fixed. See #58141
#define LLK_BOOT_BRISC 1
#include "boot.h"
#include "counters.h"

// BRISC firmware
#ifdef LLK_BOOT_MODE_BRISC

// Mailbox addresses
#ifdef COVERAGE
static const mailbox_t mailboxes_arr = reinterpret_cast<mailbox_t>(0x6DFB8U);
#else
static const mailbox_t mailboxes_arr = reinterpret_cast<mailbox_t>(0x1FFB8U);
#endif

static const mailbox_t mailbox_unpack = mailboxes_arr;
static const mailbox_t mailbox_math   = mailboxes_arr + 1;
static const mailbox_t mailbox_pack   = mailboxes_arr + 2;

static const mailbox_t profiler_barrier = reinterpret_cast<mailbox_t>(0x16AFF4U);

// A BRISC read of a TRISC's PC buffer returns once that TRISC waits on its own PC buffer read with nothing queued and its
// Tensix thread is idle, which trisc.cpp does when its kernel is done.
static const mailbox_t trisc_pc_bufs[] = {
    reinterpret_cast<mailbox_t>(PC_BUF_BASE), reinterpret_cast<mailbox_t>(PC1_BUF_BASE), reinterpret_cast<mailbox_t>(PC2_BUF_BASE)};

// One launch per release from soft reset: the host loads the kernels and releases BRISC, BRISC starts the TRISCs, waits
// on their PC buffers with no L1 traffic, then puts all four cores back in soft reset, which is what the host polls.
int main()
{
    disable_branch_prediction();

    commit_store(mailbox_math, ckernel::RESET_VAL);
    commit_store(mailbox_unpack, ckernel::RESET_VAL);
    commit_store(mailbox_pack, ckernel::RESET_VAL);

    commit_store(profiler_barrier, 0U);
    commit_store(profiler_barrier + 1, 0U);
    commit_store(profiler_barrier + 2, 0U);

    device_setup();

    llk_perf::clear_debug_features();
    // Counters on only (empty otherwise, counters.h): set up the counter banks before the TRISCs start.
    llk_perf::configure_from_brisc();

    clear_trisc_soft_reset();

#if defined(TT_METAL_TTSIM) // ttsim models no BRISC reads of the TRISC PC buffers, and nothing there is timed: wait on L1
    for (const mailbox_t mailbox : {mailbox_unpack, mailbox_math, mailbox_pack})
    {
        do
        {
            ckernel::invalidate_data_cache();
        } while (ckernel::load_blocking(mailbox) != ckernel::KERNEL_COMPLETE);
    }
#else
    for (const mailbox_t pc_buf : trisc_pc_bufs)
    {
        (void)ckernel::load_blocking(pc_buf);
    }
    // What the host waits on: polling the soft reset register instead shares the Tensix register bus with the
    // kernel's wall clock reads and moves its timings.
    *host_signal::launch_done() = ckernel::KERNEL_COMPLETE;
    while ((*host_signal::launch_done() & 0xFFU) != ckernel::KERNEL_COMPLETE)
    {
    }
#endif

    ckernel::reg_write(RISCV_DEBUG_REG_SOFT_RESET_0, ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0) | TRISC_SOFT_RESET_MASK | BRISC_SOFT_RESET_MASK);
    for (;;)
    {
    } // the write above stops this core
}

#else

int main()
{
}

#endif

extern "C" __attribute__((section(".init"), naked, noreturn)) std::uint32_t _start()
{
    do_crt0();

    main();

    for (;;)
    {
    } // Loop forever
}
