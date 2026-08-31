// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Receiver end of the ethernet wall-clock sync. See eth_wallclock_sync.hpp.

#include "tools/profiler/sync/eth_wallclock_sync.hpp"

constexpr uint32_t kNumSamples = get_compile_time_arg_val(0);
constexpr uint32_t kTimeoutLo = get_compile_time_arg_val(1);
constexpr uint32_t kTimeoutHi = get_compile_time_arg_val(2);
constexpr uint32_t kResultAddr = get_compile_time_arg_val(3);
constexpr uint32_t kChannelAddr = get_compile_time_arg_val(4);
constexpr uint32_t kHandshakeAddr = get_compile_time_arg_val(5);
constexpr uint32_t kGapCycles = get_compile_time_arg_val(6);
// RESIDENT mode: 0 = today's one-shot (run once, exit). 1 = stay resident and run one measurement burst
// per GO command written into the L1 mailbox -- one launch, many rounds, no further LaunchProgram. The
// loop is bounded two ways (EXIT command, idle-spin cap), same discipline as every other wait here.
constexpr uint32_t kResident = get_compile_time_arg_val(7);
constexpr uint32_t kMailboxAddr = get_compile_time_arg_val(8);
constexpr uint32_t kIdleSpinLo = get_compile_time_arg_val(9);
constexpr uint32_t kIdleSpinHi = get_compile_time_arg_val(10);

void kernel_main() {
    const uint64_t timeout = (static_cast<uint64_t>(kTimeoutHi) << 32) | kTimeoutLo;
    if constexpr (kResident == 0) {
        tt::tt_metal::eth_sync::eth_wallclock_sync_receiver(
        kResultAddr, kChannelAddr, kHandshakeAddr, kNumSamples, timeout);
        return;
    }
    volatile uint32_t* mbox = reinterpret_cast<volatile uint32_t*>(kMailboxAddr);
    *mbox = 0;
    const uint64_t spin_cap = (static_cast<uint64_t>(kIdleSpinHi) << 32) | kIdleSpinLo;
    uint64_t spins = 0;
    while (spins < spin_cap) {
        invalidate_l1_cache();
        const uint32_t c = *mbox;
        if (c == 2u) {
            break;  // host said exit
        }
        if (c == 1u) {
            *mbox = 0;
            tt::tt_metal::eth_sync::eth_wallclock_sync_receiver(
        kResultAddr, kChannelAddr, kHandshakeAddr, kNumSamples, timeout);
        }
        spins++;
    }
    *mbox = 0xD00DD00Du;  // exited marker the host polls for before reaping the program
}
