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
// Port 0 goes through RMWCIB, port 1 goes through SETC16.
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
        if (port == 1) TT_SETC16(a, v & mask & 0xFFFF);
        else
        {
            const std::uint32_t cur = ckernel::cfg_read(a);
            ckernel::cfg_write(a, (cur & ~mask) | (v & mask));
        }
    }
}

// Addr mods have to be restored. This is ThreadConfig, so each thread has its own values.
// Plan format is [magic][N][has_ch1x][data], where data is N x [addr32, v_t0, v_t1, v_t2],
// and, if has_ch1x is set, [adc_ch1x_unpacker][adc_ch1x_packer] at the very end.
// Most kernels don't touch tile dimension config, so (...).

// Per-thread addr-mod restore: ThreadConfig is banked per-thread (each of the 3 TRISCs compiles
// to a separate binary and owns a separate bank), so a real captured addr-mod snapshot has up to
// 3 genuinely different values per address -- unlike apply_plan_at()'s single shared value applied
// identically to every thread. Plan @ 0x1C000: [magic][N][has_ch1x] then N x [addr32, v_thread0,
// v_thread1, v_thread2], THEN, only if has_ch1x, 2 trailing words [adc_ch1x_unpacker,
// adc_ch1x_packer] -- address_counters' channel1-X ("the tile X dimension",
// cunpack_common.h/cpack_common.h) is hardware state completely outside Config[]/ThreadConfig[],
// deliberately left untouched by unpacker_addr_counter_init()/packer_addr_counter_init() (their
// own BitMask 0b1011 skips it) because a real hardware reset normally handles it; restore-mode
// never resets, so a victim can inherit an unrelated prior kernel's channel1-X instead of the
// polluter's. has_ch1x is required (not inferred from L1 leftovers) because this buffer is reused
// across trials and unwritten L1 past a shorter plan's length is stale data, not zero -- reading
// it unconditionally would plant garbage into hardware ADC state on every trial that didn't
// capture a real ch1x value. First attempt at fixing this used a SEPARATE, never-before-used L1
// address (0x1D000) and hung all three threads outright; every address that has actually worked
// this session (0x1A000, this 0x1C000 buffer) got there by extending an already-proven-safe
// buffer, never by picking a fresh one cold, so this reuses that same buffer instead of
// allocating new scratch space. Each compiled TRISC applies only its own v_thread slot (SETC16)
// and, if it's UNPACK/PACK and has_ch1x is set, its own trailing channel1-X word (SETADCXY,
// BitMask 0b0100 = channel1-X only -- see the blanket-reset regression of pack_untilize__199-203
// that motivated narrowing this to just the one field). No-op unless the host wrote the magic;
// applied after apply_plan_at()'s restore plan so a real captured value overwrites that plan's
// force-zero default.
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
    // unpacker_addr_counter_init()'s own comment: "unpacker 0, channel 1, X" specifically -- not
    // unpacker 1 -- so restore only UNP_A, matching that documented scope.
    TT_SETADCXY(ckernel::p_setadc::UNP_A, 0, plan[3 + 4 * n + 0], 0, 0, 0b0100);
#elif defined(LLK_TRISC_PACK)
    TT_SETADCXY(ckernel::p_setadc::PAC, 0, plan[3 + 4 * n + 1], 0, 0, 0b0100);
#endif
}

// ADC channel1-X restore: attempted and REVERTED (2026-09-26). address_counters is hardware state
// completely outside Config[]/ThreadConfig[] -- unpacker_addr_counter_init()/
// packer_addr_counter_init() (cunpack_common.h/cpack_common.h) deliberately exclude channel1-X from
// their own reset (BitMask 0b1011, not 0b1111), relying on a real hardware reset to zero it --
// restore-mode never triggers one, so a victim can inherit whatever channel1-X value some unrelated
// prior kernel left. Two attempts so far, both hardware-confirmed unsafe:
//   1. Blanket-set ALL four ADC channels to 0 (BitMask 0b1111): regressed victims whose own init
//      manages channel0/Y1 relative to whatever's already there -- stomping those before the
//      kernel's own init runs broke them worse than the original gap.
//   2. Touch only channel1-X (BitMask 0b0100) with the polluter's real captured value, via
//      TT_SETADCXY (the runtime-register variant, needed since the value comes from an L1 plan, not
//      a compile-time constant): this variant (`instrn_buffer[0] = ENCODING`, vs TTI_SETADCXY's
//      `.ttinsn` immediate) is not used anywhere else in this codebase, and hung all three threads
//      even with the restore plan unarmed (i.e. before ever reaching the SETADCXY call) -- root
//      cause not yet isolated; could be the unproven TT_ issuance path, or the chosen L1 scratch
//      address (0x1D000, never otherwise validated as free/safe, unlike 0x1A000/0x1C000).
// Needs the correct write mechanism confirmed (ideally via the RTL question-relay channel) before
// trying again on hardware -- not a guess-and-check target.

static inline void apply_restore_plan()
{
    apply_plan_at(LLK_RESTORE_PLAN_BASE, LLK_RESTORE_PLAN_MAGIC); // captured CFG residue
    apply_addrmod_restore();                                      // real per-thread addr-mod residue
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
        apply_restore_plan(); // config-pollution pair sweep; no-op unless host wrote a plan @ 0x1A000
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
