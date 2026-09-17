// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The sending end of the streaming profiler's link sync as a resident kernel: opens the 1588 session, handshakes
// with the receiver, then steps the link until the host writes the stop word (eth_ptp_link.hpp). Runtime args:
// the link's L1 (pilot, frame slots, control words) and the round period in refclk ticks.

#include <cstdint>

#include "eth_l1_address_map.h"
#include "internal/ethernet/dataflow_api.h"
#include "tools/profiler/sync/eth_ptp_link.hpp"

namespace eth_ptp = tt::tt_metal::eth_ptp;

static eth_ptp::SenderLink<> g_link;
static constexpr uint32_t kHandshake = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

void kernel_main() {
    const uint32_t link_l1 = get_arg_val<uint32_t>(0);
    const uint32_t pace_ticks = get_arg_val<uint32_t>(1);
    const uint32_t ctl = link_l1 + eth_ptp::kCtlOffset;
    g_link.open();
    eth_send_bytes(kHandshake, kHandshake, 16);
    eth_wait_for_receiver_done();
    g_link.start(link_l1, ctl, pace_ticks);
    volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl);
    while (*c != eth_ptp::kCtlStop) {
        g_link.step();
    }
    g_link.stop();
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl + 4) = 1;
}
