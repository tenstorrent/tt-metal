// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
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

#ifdef ARCH_WORMHOLE
#define ARCH_CYCLE_MICRO_SECOND 1000
#endif
#ifdef ARCH_BLACKHOLE
#define ARCH_CYCLE_MICRO_SECOND 1350
#endif

static const mailbox_t mailbox_unpack = mailboxes_arr;
static const mailbox_t mailbox_math   = mailboxes_arr + 1;
static const mailbox_t mailbox_pack   = mailboxes_arr + 2;

static const mailbox_t brisc_command_buffer = mailboxes_arr + 3; // 2 entries
static const mailbox_t brisc_counter        = mailboxes_arr + 5;

static const mailbox_t brisc_bread0 = mailboxes_arr + 6;
static const mailbox_t brisc_bread1 = mailboxes_arr + 7;

static const mailbox_t profiler_barrier = reinterpret_cast<mailbox_t>(0x16AFF4U);

enum class BriscCommandState : std::uint32_t
{
    IDLE_STATE                        = 0,
    START_TRISCS                      = 1,
    RESET_TRISCS                      = 2,
    UPDATE_START_ADDR_CACHE_AND_START = 3,
    // Wormhole perf launches: as 1 and 3, then serve the TRISC barrier (see dbg_barrier)
    START_TRISCS_DBG_BARRIER                      = 4,
    UPDATE_START_ADDR_CACHE_AND_START_DBG_BARRIER = 5,
};

// Written to `brisc_counter` as the LAST step of firmware init. The host polls
// for this value after deasserting BRISC reset to confirm the firmware has
// finished init and entered the polling loop. Required for read-pumped sim
// targets (TTSim) where host writes alone do not advance the clock — without
// this handshake, the host's first command write can race with the firmware's
// own zero-init of the command slots. The sentinel is overwritten by the
// regular protocol counter as soon as the first command is processed.
constexpr std::uint32_t BRISC_BOOT_READY_SENTINEL = 0xB001CAFEU;

#if defined(ARCH_WORMHOLE) && !defined(TT_METAL_TTSIM) // serves the TRISC rendezvous of barrier.h park() in perf launches
namespace dbg_barrier
{
constexpr std::uint32_t PC_BUF[3]        = {0xFFE80000U, 0xFFE90000U, 0xFFEA0000U};
constexpr std::uint32_t DBG_CNTL_0       = 0xFFB12080U;
constexpr std::uint32_t DBG_CNTL_1       = 0xFFB12084U;
constexpr std::uint32_t DBG_STATUS_0     = 0xFFB12088U;
constexpr std::uint32_t DBG_STATUS_1     = 0xFFB1208CU;
constexpr std::uint32_t REQ              = 1U << 31;
constexpr std::uint32_t READ_VALID       = 1U << 30;
constexpr std::uint32_t WR               = 1U << 16;
constexpr std::uint32_t REG_STATUS       = 0;
constexpr std::uint32_t REG_COMMAND      = 1;
constexpr std::uint32_t STATUS_PAUSED    = 1U << 0;
constexpr std::uint32_t COMMAND_CONTINUE = 1U << 2;
constexpr std::uint32_t COMMAND_FLUSH    = 1U << 8; // without debug mode (bit 31) the flush restarts at the reset PC

inline volatile std::uint32_t& reg(std::uint32_t addr)
{
    return *reinterpret_cast<volatile std::uint32_t*>(addr);
}

constexpr std::uint32_t DBG_REGS        = DBG_CNTL_0 - 0x80U;
constexpr std::uint32_t PC_BUF_OVERRIDE = 0xFFB12090U; // TRISC_PC_BUF_OVERRIDE: bit 10 * t = override enable of TRISC t
constexpr std::uint32_t RELEASE_ALL     = (1U << 0) | (1U << 10) | (1U << 20);
static_assert(REQ == 1U << 31 && READ_VALID == 1U << 30 && REG_STATUS == 0, "serve() tests REQ and READ_VALID by sign");
static_assert(DBG_CNTL_1 == DBG_CNTL_0 + 4 && DBG_STATUS_0 == DBG_CNTL_0 + 8 && DBG_STATUS_1 == DBG_CNTL_0 + 12 && PC_BUF_OVERRIDE == DBG_CNTL_0 + 16);

// Serve loop, one rendezvous per pass: wait until all TRISCs park, let them halt, flush them and release them together.
// Pinned in assembly (GCC's code for the C version) in its own 1 KiB section, so no other BRISC change moves its timing.
__attribute__((noinline, noipa, section(".text.llk_dbg_serve"), aligned(1024))) void serve()
{
    asm volatile(
        "li    t0, 0x00ffffff\n\t"
        "li    t4, %[wr_cmd]\n\t"
        "li    t3, %[req_wr_cmd]\n\t"
        "li    t1, %[pcb0]\n\t"
        "li    a7, %[pcb1]\n\t"
        "li    a6, %[pcb2]\n\t"
        "li    s0, %[slot0]\n\t"
        "li    t2, %[complete]\n\t"
        "li    s2, %[slot1]\n\t"
        "li    s1, %[slot2]\n\t"
        "li    t6, %[req]\n\t"
        "li    a5, %[dbg]\n\t"
        "li    a0, %[tstep]\n\t"
        "li    a1, %[tend]\n\t"
        "li    t5, %[flush_cmd]\n\t"
        "li    s4, %[release]\n"
        "1:\n\t" // arrive
        "lw    a4, 0(t1)\n\tand   a4, a4, a4\n\t"
        "lw    a4, 0(a7)\n\tand   a4, a4, a4\n\t"
        "lw    a4, 0(a6)\n\tand   a4, a4, a4\n\t"
        "lw    a4, %[scratch](s0)\n\tand   a4, a4, t0\n\tbeq   a4, t2, 9f\n"
        "2:\n\t"
        "sw    zero, 0(t1)\n\tsw    zero, 0(a7)\n\tsw    zero, 0(a6)\n\t" // on to the ebreak
        // halted? The debug request bit is synchronized and edge detected (tt_tensix.sv): hold it until STATUS_0
        // shows it, then let it fall
        "lui   a2, %%hi(%[tstep])\n"
        "3:\n\t"
        "or    s3, a2, t6\n"
        "4:\n\t"
        "sw    s3, %[cntl0](a5)\n"
        "5:\n\t"
        "lw    a4, %[status0](a5)\n\tbgez  a4, 5b\n\t"
        "sw    a2, %[cntl0](a5)\n"
        "6:\n\t"
        "lw    a4, %[status0](a5)\n\tbltz  a4, 6b\n"
        "7:\n\t"
        "lw    a4, %[status0](a5)\n\tslli  a3, a4, 1\n\tbgez  a3, 7b\n\t" // read valid
        "lw    a4, %[status1](a5)\n\tandi  a4, a4, %[paused]\n\tbeqz  a4, 4b\n\t"
        "add   a2, a2, a0\n\tbne   a2, a1, 3b\n\t"
        "li    a4, 512\n" // fetch ahead of the halted cores settles
        "8:\n\t"
        "nop\n\taddi  a4, a4, -1\n\tbnez  a4, 8b\n\t"
        "li    a3, 1024\n" // 64 L1 reads from address 0 (a4 = 0): every bank arbiter last granted BRISC
        "10:\n\t"
        "lw    a2, 0(a4)\n\tandi  a2, a2, 0\n\taddi  a4, a4, 16\n\tbne   a4, a3, 10b\n\t"
        "lui   a4, %%hi(%[icinv])\n\tli    a3, 14\n\tsw    a3, %%lo(%[icinv])(a4)\n\t" // TRISC 0-2 icaches
        "li    a4, 64\n"
        "11:\n\t"
        "nop\n\taddi  a4, a4, -1\n\tbnez  a4, 11b\n\t"
        "lui   a3, %%hi(%[tstep])\n\t" // flush: COMMAND = FLUSH | CONTINUE, TRISC 0, 1, 2
        "addi  s3, a5, %[cntl1]\n"
        "12:\n\t"
        "sw    t5, 0(s3)\n\t"
        "or    a4, a3, t3\n\t"
        "sw    a4, %[cntl0](a5)\n\t"
        "or    a2, a3, t4\n"
        "13:\n\t"
        "lw    a4, %[status0](a5)\n\tbgez  a4, 13b\n\t"
        "sw    a2, %[cntl0](a5)\n"
        "14:\n\t"
        "lw    a4, %[status0](a5)\n\tbltz  a4, 14b\n\t"
        "add   a3, a3, a0\n\tbne   a3, a1, 12b\n\t"
        "lw    a4, 0(t1)\n\tand   a4, a4, a4\n\t" // all parked on their hold read: nothing below depends on data
        "lw    a4, 0(a7)\n\tand   a4, a4, a4\n\t"
        "lw    a4, 0(a6)\n\tand   a4, a4, a4\n\t"
        "li    a4, 512\n"
        "15:\n\t"
        "nop\n\taddi  a4, a4, -1\n\tbnez  a4, 15b\n\t"
        "sw    zero, 0(t1)\n\tsw    zero, 0(a7)\n\tsw    zero, 0(a6)\n\t" // release: unpack, math, pack back to back
        "j     1b\n"
        "9:\n\t" // TRISC 0 is done: done if all are, else serve
        "lw    a4, %[scratch](s2)\n\tand   a4, a4, t0\n\tbne   a4, t2, 2b\n\t"
        "lw    a4, %[scratch](s1)\n\tand   a4, a4, t0\n\tbne   a4, t2, 2b\n\t"
        "sw    zero, 0(t1)\n\tsw    zero, 0(a7)\n\tsw    zero, 0(a6)\n\t" // out of the kernel
        :
        : [pcb0] "i"(PC_BUF[0]),
          [pcb1] "i"(PC_BUF[1]),
          [pcb2] "i"(PC_BUF[2]),
          [slot0] "i"(host_signal::NOC_OVERLAY_START_ADDR),
          [slot1] "i"(host_signal::NOC_OVERLAY_START_ADDR + host_signal::NOC_STREAM_REG_SPACE_SIZE),
          [slot2] "i"(host_signal::NOC_OVERLAY_START_ADDR + 2 * host_signal::NOC_STREAM_REG_SPACE_SIZE),
          [scratch] "i"(host_signal::STREAM_SCRATCH_REG_INDEX * 4),
          [complete] "i"(ckernel::KERNEL_COMPLETE & 0xFFFFFFU),
          [req] "i"(REQ),
          [dbg] "i"(DBG_REGS),
          [cntl0] "i"(DBG_CNTL_0 - DBG_REGS),
          [cntl1] "i"(DBG_CNTL_1 - DBG_REGS),
          [status0] "i"(DBG_STATUS_0 - DBG_REGS),
          [status1] "i"(DBG_STATUS_1 - DBG_REGS),
          [override] "i"(PC_BUF_OVERRIDE - DBG_REGS),
          [paused] "i"(STATUS_PAUSED),
          [tstep] "i"(1U << 17), // debug target field: TRISC t is t + 1
          [tend] "i"(4U << 17),
          [wr_cmd] "i"(WR | REG_COMMAND),
          [req_wr_cmd] "i"(REQ | WR | REG_COMMAND),
          [flush_cmd] "i"(COMMAND_FLUSH | COMMAND_CONTINUE),
          [icinv] "i"(TENSIX_CFG_BASE + 4 * RISCV_IC_INVALIDATE_InvalidateAll_ADDR32),
          [release] "i"(RELEASE_ALL)
        : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "s0", "s1", "s2", "s3", "s4", "memory");
}
} // namespace dbg_barrier
#endif

void reset_state(std::uint32_t& counter)
{
    counter++;
    // Double buffer protocol: host writes the next command to slot (counter & 1),
    // BRISC reads from the same slot. After processing, bump counter so both sides
    // move to the other slot, and zero the new slot to prevent retriggering.
    ckernel::store_blocking(brisc_command_buffer + (counter & 1), static_cast<std::uint32_t>(BriscCommandState::IDLE_STATE));
    commit_store(brisc_counter, counter);
    host_signal::write(host_signal::BRISC_COUNTER_SLOT, counter);
}

int main()
{
    disable_branch_prediction();

    std::uint32_t counter = 0;

    ckernel::store_blocking(brisc_command_buffer, 0);
    ckernel::store_blocking(brisc_command_buffer + 1, 0);
    ckernel::store_blocking(brisc_bread0, 0);
    ckernel::store_blocking(brisc_bread1, 0);

    // LAST init step: publish the boot-ready sentinel so the host can confirm
    // the firmware is in the polling loop before it issues any command. Uses
    // commit_store (store + spin-readback) for a hard visibility guarantee.
    commit_store(brisc_counter, BRISC_BOOT_READY_SENTINEL);
    host_signal::write(host_signal::BRISC_COUNTER_SLOT, BRISC_BOOT_READY_SENTINEL);

#ifdef ARCH_WORMHOLE
    // Array for keeping last known addresses of _start symbol in kernel ELF, for T[0-2]
    std::uint32_t TRISC_ADDR_CACHE[3] = {};
#endif

    while (true)
    {
        ckernel::invalidate_data_cache();

        // Poll the active slot of the double buffered command mailbox.
        // The host writes to slot (counter & 1) and BRISC reads the same slot.
        // Using load_blocking ensures the read completes before the switch.
        const auto command = static_cast<BriscCommandState>(ckernel::load_blocking(brisc_command_buffer + (counter & 1)));
#if defined(ARCH_WORMHOLE)
        [[maybe_unused]] const bool perf_barrier =
            command == BriscCommandState::START_TRISCS_DBG_BARRIER || command == BriscCommandState::UPDATE_START_ADDR_CACHE_AND_START_DBG_BARRIER;
#endif
        switch (command)
        {
            // Wormhole specific, on Blackhole this command has same behaviour as BriscCommandState::START_TRISCS
#if defined(ARCH_WORMHOLE)
            case BriscCommandState::UPDATE_START_ADDR_CACHE_AND_START_DBG_BARRIER:
#endif
            case BriscCommandState::UPDATE_START_ADDR_CACHE_AND_START:
#ifdef ARCH_WORMHOLE
                // Elf loader can't put T[0-2] PCs to point to _start addresses of every ELF. Thus host needs to write them at particular location,
                // in case of LLK testing infra, that is last 12 bytes of L1, for T[0-2] to read from right after it's released from reset. Side-effect
                // of this action(s) is that T[0-2] reset these locations after they read them for this purpose. Because of this, when host loads new ELFs
                // it needs to tell BRISC to cache those values again, which this block of code does. Afterwards it proceeds with regular kernel start sequence
                for (int i = 0; i < 3; i++)
                {
                    TRISC_ADDR_CACHE[i] = ckernel::load_blocking(trisc_start_addresses + i);
                }
#endif
                [[fallthrough]];
#if defined(ARCH_WORMHOLE)
            case BriscCommandState::START_TRISCS_DBG_BARRIER:
#endif
            case BriscCommandState::START_TRISCS:

#ifdef ARCH_WORMHOLE
                // Load cached addresses of _start symbol of every kernel ELF is case of Wormhole
                for (int i = 0; i < 3; i++)
                {
                    commit_store(trisc_start_addresses + i, TRISC_ADDR_CACHE[i]);
                }
#endif

                commit_store(mailbox_math, ckernel::RESET_VAL);
                commit_store(mailbox_unpack, ckernel::RESET_VAL);
                commit_store(mailbox_pack, ckernel::RESET_VAL);
                for (std::uint32_t slot = 0; slot < 3; ++slot)
                {
                    host_signal::write(slot, ckernel::RESET_VAL);
                }

                commit_store(profiler_barrier, 0U);
                commit_store(profiler_barrier + 1, 0U);
                commit_store(profiler_barrier + 2, 0U);

                device_setup();

                // Configure + arm counters before releasing TRISCs (no-op in NC builds).
                llk_perf::configure_and_arm_from_brisc();

                clear_trisc_soft_reset();

                reset_state(counter);
                commit_store(brisc_bread0, counter);
#if defined(ARCH_WORMHOLE) && !defined(TT_METAL_TTSIM)
                if (perf_barrier)
                {
                    dbg_barrier::serve();
                }
#endif
                break;

            case BriscCommandState::RESET_TRISCS:
                set_triscs_soft_reset();

                reset_state(counter);
                commit_store(brisc_bread1, counter);
                break;

            default:
                break;
        }

#if defined(TT_METAL_TTSIM) // ttsim simulates every NOP and nothing there interferes, so it polls every microsecond
        constexpr std::uint32_t poll_period_us = 1;
#else
        constexpr std::uint32_t poll_period_us = 100;
#endif
        // Poll about every 100 us and spin on NOPs in between: each poll is an L1 read, and the old wall clock wait kept
        // the debug register bus busy. Both changed the timing of the kernel under test.
        for (std::uint32_t i = 0; i < poll_period_us * ARCH_CYCLE_MICRO_SECOND; ++i)
        {
            asm volatile("nop");
        }
    }
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
