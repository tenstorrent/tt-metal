// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sender end of the ethernet wall-clock sync. See eth_wallclock_sync.hpp.

#include "tools/profiler/sync/eth_wallclock_sync.hpp"

constexpr uint32_t kNumSamples = get_compile_time_arg_val(0);
constexpr uint32_t kTimeoutLo = get_compile_time_arg_val(1);
constexpr uint32_t kTimeoutHi = get_compile_time_arg_val(2);
constexpr uint32_t kResultAddr = get_compile_time_arg_val(3);
constexpr uint32_t kChannelAddr = get_compile_time_arg_val(4);
constexpr uint32_t kHandshakeAddr = get_compile_time_arg_val(5);
constexpr uint32_t kGapCycles = get_compile_time_arg_val(6);

void kernel_main() {
    const uint64_t timeout = (static_cast<uint64_t>(kTimeoutHi) << 32) | kTimeoutLo;
    tt::tt_metal::eth_sync::eth_wallclock_sync_sender(
        kResultAddr, kChannelAddr, kHandshakeAddr, kNumSamples, timeout, kGapCycles);
}
