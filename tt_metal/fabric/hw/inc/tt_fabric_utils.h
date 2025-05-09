// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eth_chan_noc_mapping.h"
#include "tt_metal/hw/inc/ethernet/tunneling.h"
#include "tt_metal/hw/inc/dataflow_api.h"
#include "tt_metal/hw/inc/dataflow_api_addrgen.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"

namespace tt::tt_fabric {

/* Termination signal handling*/
FORCE_INLINE bool got_immediate_termination_signal(volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    return *termination_signal_ptr == tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE;
}
FORCE_INLINE bool got_graceful_termination_signal(volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    return *termination_signal_ptr == tt::tt_fabric::TerminationSignal::GRACEFULLY_TERMINATE;
}
FORCE_INLINE bool got_termination_signal(volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    return got_immediate_termination_signal(termination_signal_ptr) ||
           got_graceful_termination_signal(termination_signal_ptr);
}

FORCE_INLINE bool connect_is_requested(uint32_t cached) {
    return cached == tt::tt_fabric::EdmToEdmSender<0>::open_connection_value ||
           cached == tt::tt_fabric::EdmToEdmSender<0>::close_connection_request_value;
}

// !!!FORCE_INLINE could potentially cause stack corruption as seen in the past
inline void wait_for_notification(uint32_t address, uint32_t value) {
    volatile tt_l1_ptr uint32_t* poll_addr = (volatile tt_l1_ptr uint32_t*)address;
    while (*poll_addr != value) {
        // context switch while waiting to allow slow dispatch traffic to go through
        run_routing();
    }
}

// !!!FORCE_INLINE could potentially cause stack corruption as seen in the past
inline void notify_master_router(uint32_t master_eth_chan, uint32_t address) {
    // send semaphore increment to master router on this device.
    // semaphore notifies all other routers that this router has completed
    // startup handshake with its ethernet peer.
    uint64_t dest_addr = get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][master_eth_chan], address);
    noc_fast_atomic_increment<DM_DEDICATED_NOC, true>(
        noc_index,
        NCRISC_AT_CMD_BUF,
        dest_addr,
        NOC_UNICAST_WRITE_VC,
        1,
        31,
        false,
        false,
        MEM_NOC_ATOMIC_RET_VAL_ADDR);
}

// !!!FORCE_INLINE could potentially cause stack corruption as seen in the past
inline void notify_slave_routers(
    uint32_t router_eth_chans_mask, uint32_t master_eth_chan, uint32_t address, uint32_t notification) {
    uint32_t remaining_cores = router_eth_chans_mask;
    for (uint32_t i = 0; i < 16; i++) {
        if (remaining_cores == 0) {
            break;
        }
        if ((remaining_cores & (0x1 << i)) && (master_eth_chan != i)) {
            uint64_t dest_addr = get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], address);
            noc_inline_dw_write(dest_addr, notification);
            remaining_cores &= ~(0x1 << i);
        }
    }
}

}  // namespace tt::tt_fabric
