// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The sending end of the streaming profiler's link sync as a resident kernel: opens the 1588 session, handshakes
// with the receiver, then steps the link until the host writes the stop word (eth_ptp_link.hpp). Runtime arg: the
// link's L1, a kernel_profiler::LinkSyncL1.

#include <cstdint>

#include "eth_l1_address_map.h"
#include "internal/ethernet/dataflow_api.h"
#include "tools/profiler/sync/eth_ptp_link.hpp"

namespace eth_ptp = tt::tt_metal::eth_ptp;

static eth_ptp::SenderLink<> g_link;
static constexpr uint32_t kHandshake = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

void kernel_main() {
    const uint32_t link_l1 = get_arg_val<uint32_t>(0);
    volatile eth_ptp::LinkL1* l1 = reinterpret_cast<volatile eth_ptp::LinkL1*>(link_l1);
    g_link.open(link_l1);
    eth_send_bytes(kHandshake, kHandshake, 16);
    eth_wait_for_receiver_done();
    g_link.start();
    while (l1->ctl != eth_ptp::kCtlStop) {
        g_link.step();
    }
    g_link.stop();
    l1->done = 1;
}
