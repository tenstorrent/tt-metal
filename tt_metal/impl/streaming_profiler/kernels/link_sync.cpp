// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// When fabric isn't running, the link sync runs as this kernel, on each port's eth core.

#include <cstdint>

#include "eth_l1_address_map.h"
#include "internal/ethernet/dataflow_api.h"
#include "tt_metal/impl/streaming_profiler/kernels/link_sync.hpp"

constexpr auto kRole = static_cast<kernel_profiler::LinkSyncRole>(get_named_compile_time_arg_val("LINK_SYNC_ROLE"));
static link_sync::Port<kRole> g_port;
constexpr uint32_t kHandshake = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
constexpr uint32_t kHandshakeBytes = 16;
// How many loops the kernel runs between context switches to the Ethernet firmware, which shares ERISC0 and only runs
// when a kernel switches to it. The interval matches the fabric router's default_firmware_context_switch_interval.
constexpr uint32_t kLoopsPerContextSwitch = 10000;

void kernel_main() {
    g_port.start();
    // The transmitter waits until the receiver has installed its RX stamp rule, so every frame gets an ingress stamp.
    if constexpr (kRole == kernel_profiler::LinkSyncRole::Transmitter) {
        eth_send_bytes(kHandshake, kHandshake, kHandshakeBytes);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(kHandshakeBytes);
        eth_receiver_channel_done(0);
    }
    uint32_t until_switch = kLoopsPerContextSwitch;
    // The receiver's due() and the transmitter's serve() refresh the L1 cache, so this read sees the host's Stop.
    while (g_port.ctl() != kernel_profiler::LinkSyncCtl::Stop) {
        if (g_port.due()) {
            g_port.serve();
        }
        if (--until_switch == 0) {
            until_switch = kLoopsPerContextSwitch;
            run_routing();
        }
    }
    g_port.stop();
}
