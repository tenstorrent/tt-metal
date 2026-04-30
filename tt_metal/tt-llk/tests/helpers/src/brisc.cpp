// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "boot.h"

#ifdef LLK_BOOT_MODE_BRISC

// Mailbox addresses
#ifdef COVERAGE
mailbox_t mailboxes_arr = (mailbox_t)0x6DFB8U;
#else
mailbox_t mailboxes_arr = (mailbox_t)0x1FFB8U;
#endif

#ifdef ARCH_WORMHOLE
#define ARCH_CYCLE_MICRO_SECOND 1000
#endif
#ifdef ARCH_BLACKHOLE
#define ARCH_CYCLE_MICRO_SECOND 1350
#endif

mailbox_t mailbox_unpack = mailboxes_arr;
mailbox_t mailbox_math   = mailboxes_arr + 1;
mailbox_t mailbox_pack   = mailboxes_arr + 2;

mailbox_t brisc_command_buffer = mailboxes_arr + 3; // 2 entries
mailbox_t brisc_counter        = mailboxes_arr + 5;

mailbox_t brisc_bread0 = mailboxes_arr + 6;
mailbox_t brisc_bread1 = mailboxes_arr + 7;

mailbox_t profiler_barrier = (mailbox_t)0x16AFF4U;

enum class BriscCommandState : std::uint32_t
{
    IDLE_STATE                        = 0,
    START_TRISCS                      = 1,
    RESET_TRISCS                      = 2,
    UPDATE_START_ADDR_CACHE_AND_START = 3,
};

void reset_state(std::uint32_t& counter)
{
    counter++;
    ckernel::store_blocking(brisc_command_buffer + (counter & 1), static_cast<std::uint32_t>(BriscCommandState::IDLE_STATE));
    commit_store(brisc_counter, counter);
}

int main()
{
    disable_branch_prediction();

    std::uint32_t counter = 0;

    ckernel::store_blocking(brisc_command_buffer, 0);
    ckernel::store_blocking(brisc_command_buffer + 1, 0);
    ckernel::store_blocking(brisc_counter, 0);
    ckernel::store_blocking(brisc_bread0, 0);
    ckernel::store_blocking(brisc_bread1, 0);

#ifdef ARCH_WORMHOLE
    // Array for keeping last known addresses of _start symbol in kernel ELF, for T[0-2]
    std::uint32_t TRISC_ADDR_CACHE[3] = {};
#endif

    while (true)
    {
        ckernel::invalidate_data_cache();

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
