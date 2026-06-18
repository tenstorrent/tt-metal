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
barrier_ptr_t barrier_ptr   = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
buffer_ptr_t buffer         = reinterpret_cast<buffer_ptr_t>(BUFFERS_START);
std::uint32_t write_idx     = 0;
std::uint32_t open_zone_cnt = 0;

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

// === In-kernel CFG pollution (init-completeness fuzzing) ===
// Runs on EACH thread just before run_kernel (after reset_cfg_state_id, so cfg writes hit
// state-0 bank). Reads a plan the host wrote to the device-print L1 region (free unless the
// print build flag is passed -> guarded by #ifndef LLK_DEVICE_PRINT_BUFFER_BASE). Unlike host
// CFG writes, this thrashes config through the SAME ports the kernel reads: TT_SETC16 reaches
// thread-private ThreadConfig (addr-mod etc.), cfg_write reaches the shared banked space.
// Plan @ 0x15000: [magic][N] then N quads [addr32, value, port, mask] (port 1=SETC16/16b,
// 0=cfg_write/32b; mask = bits to write, rest preserved via read-modify-write on the shared port).
// No-op unless the magic is present (a fresh reset clears L1, so clean runs don't apply a stale plan).
#ifndef LLK_DEVICE_PRINT_BUFFER_BASE
static constexpr std::uint32_t LLK_POLLUTE_PLAN_BASE  = 0x15000;
static constexpr std::uint32_t LLK_POLLUTE_PLAN_MAGIC = 0x504F4C31u; // 'POL1'
// Restore plan: pristine (post-reset) CFG values the host replays BEFORE the poison so a trial
// starts from a clean baseline WITHOUT a per-trial tt-smi -r (CFG persists across launch, so a
// prior trial's poison would otherwise accumulate in the never-written fields we hunt). Same quad
// format as the poison plan; lives in the free L1 gap below buf_a (0x21000). Applied first; the
// poison plan then overlays this trial's subset. No-op unless the host wrote the magic.
static constexpr std::uint32_t LLK_RESTORE_PLAN_BASE  = 0x1A000;
static constexpr std::uint32_t LLK_RESTORE_PLAN_MAGIC = 0x52535431u; // 'RST1'

// Apply a quad-list plan [magic][N] then N x [addr32, value, port, mask] at `base`. port 1 =
// SETC16 (thread-private: addr-mod, state id), port 0 = cfg_write RMW (shared banked, state-0
// bank). mask selects which bits to write; the rest are preserved (firmware-owned bits, or
// sub-field isolation). Used for BOTH restore (mask=full word / 0xFFFF) and poison.
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
            TT_SETC16(a, v & mask & 0xFFFF); // thread-private (addr-mod, state id, ...)
        }
        else
        {
            // RMW so unmasked bits (e.g. firmware-owned DISABLE_RISC_BP) are preserved.
            const std::uint32_t cur = ckernel::cfg_read(a);
            ckernel::cfg_write(a, (cur & ~mask) | (v & mask)); // shared banked CFG (state-0 bank)
        }
    }
}

static inline void apply_pollution_plan()
{
    apply_plan_at(LLK_RESTORE_PLAN_BASE, LLK_RESTORE_PLAN_MAGIC); // pristine baseline (restore-mode)
    apply_plan_at(LLK_POLLUTE_PLAN_BASE, LLK_POLLUTE_PLAN_MAGIC); // this trial's poison overlay
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
        apply_pollution_plan(); // init-completeness fuzzing; no-op unless host wrote a plan @ 0x15000
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
