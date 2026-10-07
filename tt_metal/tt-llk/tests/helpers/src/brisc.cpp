// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <type_traits>

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

// The request bit is synchronized and edge detected (tt_tensix.sv): hold it until STATUS_0 shows it, then let it fall.
inline void request(std::uint32_t cntl)
{
    reg(DBG_CNTL_0) = cntl | REQ;
    while (!(reg(DBG_STATUS_0) & REQ))
    {
    }
    reg(DBG_CNTL_0) = cntl;
    while (reg(DBG_STATUS_0) & REQ)
    {
    }
}

inline std::uint32_t read(std::uint32_t trisc, std::uint32_t index)
{
    request(((trisc + 1) << 17) | index);
    while (!(reg(DBG_STATUS_0) & READ_VALID))
    {
    }
    return reg(DBG_STATUS_1);
}

inline void write(std::uint32_t trisc, std::uint32_t index, std::uint32_t value)
{
    reg(DBG_CNTL_1) = value;
    request(((trisc + 1) << 17) | WR | index);
}

inline bool kernel_complete(std::uint32_t trisc)
{
    const std::uint32_t v =
        reg(host_signal::NOC_OVERLAY_START_ADDR + trisc * host_signal::NOC_STREAM_REG_SPACE_SIZE + host_signal::STREAM_SCRATCH_REG_INDEX * 4);
    return (v & 0xFFFFFFU) == (ckernel::KERNEL_COMPLETE & 0xFFFFFFU);
}

// A BRISC read of a PC buffer returns once that TRISC is blocked on its word 0 read and idle, so this waits without polling.
inline void wait_all_parked()
{
    for (std::uint32_t t = 0; t < 3; ++t)
    {
        (void)ckernel::load_blocking(reinterpret_cast<volatile std::uint32_t*>(PC_BUF[t]));
    }
}

inline void release_all()
{
    for (std::uint32_t t = 0; t < 3; ++t)
    {
        reg(PC_BUF[t]) = 0;
    }
}

inline void spin(std::uint32_t n)
{
    for (std::uint32_t i = 0; i < n; ++i)
    {
        asm volatile("nop");
    }
}

void serve()
{
    for (;;)
    {
        wait_all_parked();
        const bool done = kernel_complete(0) && kernel_complete(1) && kernel_complete(2);
        release_all(); // on to the ebreak, or out of the kernel
        if (done)
        {
            return;
        }
        for (std::uint32_t t = 0; t < 3; ++t)
        {
            while (!(read(t, REG_STATUS) & STATUS_PAUSED))
            {
            }
        }
        spin(512);                                               // fetch ahead of the halted cores has settled
        for (std::uint32_t addr = 0; addr < 64 * 16; addr += 16) // every L1 bank arbiter last granted BRISC (asm: address 0 is valid L1)
        {
            std::uint32_t v;
            asm volatile("lw %0, 0(%1)\n\tandi %0, %0, 0" : "=r"(v) : "r"(addr) : "memory");
        }
        reinterpret_cast<volatile std::uint32_t*>(TENSIX_CFG_BASE)[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] = 0b1110;
        spin(64);
        for (std::uint32_t t = 0; t < 3; ++t)
        {
            write(t, REG_COMMAND, COMMAND_FLUSH | COMMAND_CONTINUE);
        }
        wait_all_parked();
        spin(512);
        release_all();
    }
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

#if defined(TT_METAL_TTSIM) || defined(LLK_SIMULATOR) // simulators run every NOP; poll every microsecond
        constexpr std::uint32_t poll_period_us = 1;
#else
        constexpr std::uint32_t poll_period_us = 100;
#endif
        // Poll about every 100 us and spin on NOPs in between: each poll is an L1 read, and the old wall clock wait kept
        // the debug register bus busy. Both changed the timing of the kernel under test.
#if defined(LLK_BRISC_OLD_POLL) // experiment: the poll before e02b20f56ee, every 1 us on the wall clock
        (void)poll_period_us;
        ckernel::wait(ARCH_CYCLE_MICRO_SECOND);
#else
        for (std::uint32_t i = 0; i < poll_period_us * ARCH_CYCLE_MICRO_SECOND; ++i)
        {
            asm volatile("nop");
        }
#endif
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
