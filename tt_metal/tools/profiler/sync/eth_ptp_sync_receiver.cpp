// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/sync/eth_ptp_sync_kernel.hpp"

constexpr uint32_t kNumSamples = get_compile_time_arg_val(0);
constexpr uint32_t kTimeoutLo = get_compile_time_arg_val(1);
constexpr uint32_t kTimeoutHi = get_compile_time_arg_val(2);
constexpr uint32_t kResultAddr = get_compile_time_arg_val(3);
constexpr uint32_t kChannelAddr = get_compile_time_arg_val(4);
constexpr uint32_t kHandshakeAddr = get_compile_time_arg_val(5);
constexpr uint32_t kPtpAddr = get_compile_time_arg_val(7);
constexpr uint32_t kFlags = get_compile_time_arg_val(8);

void kernel_main() {
    const uint64_t timeout = (static_cast<uint64_t>(kTimeoutHi) << 32) | kTimeoutLo;
    tt::tt_metal::eth_ptp::ptp_sync_receiver(
        kResultAddr, kPtpAddr, kChannelAddr, kHandshakeAddr, kNumSamples, timeout, kFlags);
}
