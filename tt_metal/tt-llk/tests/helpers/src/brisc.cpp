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
};

// Written to `brisc_counter` as the LAST step of firmware init. The host polls
// for this value after deasserting BRISC reset to confirm the firmware has
// finished init and entered the polling loop. Required for read-pumped sim
// targets (TTSim) where host writes alone do not advance the clock — without
// this handshake, the host's first command write can race with the firmware's
// own zero-init of the command slots. The sentinel is overwritten by the
// regular protocol counter as soon as the first command is processed.
constexpr std::uint32_t BRISC_BOOT_READY_SENTINEL = 0xB001CAFEU;

void reset_state(std::uint32_t& counter)
{
    counter++;
    // Double buffer protocol: host writes the next command to slot (counter & 1),
    // BRISC reads from the same slot. After processing, bump counter so both sides
    // move to the other slot, and zero the new slot to prevent retriggering.
    ckernel::store_blocking(brisc_command_buffer + (counter & 1), static_cast<std::uint32_t>(BriscCommandState::IDLE_STATE));
    commit_store(brisc_counter, counter);
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
        switch (static_cast<BriscCommandState>(ckernel::load_blocking(brisc_command_buffer + (counter & 1))))
        {
            // Wormhole specific, on Blackhole this command has same behaviour as BriscCommandState::START_TRISCS
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

                commit_store(profiler_barrier, 0U);
                commit_store(profiler_barrier + 1, 0U);
                commit_store(profiler_barrier + 2, 0U);

                device_setup();

                // Configure + arm counters before releasing TRISCs (no-op in NC builds).
                llk_perf::configure_and_arm_from_brisc();

                clear_trisc_soft_reset();

                reset_state(counter);
                commit_store(brisc_bread0, counter);
                break;

            case BriscCommandState::RESET_TRISCS:
                set_triscs_soft_reset();

                reset_state(counter);
                commit_store(brisc_bread1, counter);
                break;

            default:
                break;
        }

        // Wait for 1us before polling again
        ckernel::wait(ARCH_CYCLE_MICRO_SECOND);
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
