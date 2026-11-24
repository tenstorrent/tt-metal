// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// this is needed for inclusion of fabric_erisc_datamover_channels.hpp, since we are not
// including fabric_erisc_router_ct_args.hpp here, where the constant is originally defined
namespace tt::tt_fabric {
static constexpr uint8_t worker_handshake_noc = noc_index;
}  // namespace tt::tt_fabric

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "hostdev/fabric_edm_packet_header.hpp"

namespace tt::tt_fabric {

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS>
using FabricMuxChannelBuffer = EthChannelBuffer<PACKET_HEADER_TYPE, FABRIC_MUX_CHANNEL_NUM_BUFFERS>;

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS>
using FabricMuxStaticSizedChannelWorkerInterface =
    StaticSizedSenderChannelWorkerInterface<tt::tt_fabric::worker_handshake_noc, FABRIC_MUX_CHANNEL_NUM_BUFFERS>;

using FabricMuxChannelClientLocationInfo = EDMChannelWorkerLocationInfo;

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS>
using WorkerToFabricMuxSender = WorkerToFabricEdmSenderImpl<false, FABRIC_MUX_CHANNEL_NUM_BUFFERS>;

using FabricMuxStatus = EDMStatus;

// Because the mux to producer (worker) (ack) path uses counters, we initialize our EdmChannelWorkerInterface
// (the interface from mux ack to worker) to start at 0. **Note** that if this is ever updated to be free slots
// based instead, we'd initialize to num buffer slots (NUM_EDM_BUFFERS) instead.
constexpr size_t MUX_TO_WORKER_INTERFACE_STARTING_READ_COUNTER_VALUE = 0;

// A mux in program-local L1 publishes the L1 address of its memory map, which is never 0, in a program semaphore that
// starts at 0 at every launch. Polls that semaphore on the mux at (mux_noc_x, mux_noc_y) until the mux publishes, and
// returns the address. local_scratch_address is a word of this core's L1 to read into.
FORCE_INLINE uint32_t poll_fabric_mux_l1_base(
    uint8_t mux_noc_x, uint8_t mux_noc_y, uint32_t l1_base_semaphore_id, uint32_t local_scratch_address, uint8_t noc) {
    const uint64_t l1_base_noc_addr = get_noc_addr(mux_noc_x, mux_noc_y, get_semaphore(l1_base_semaphore_id), noc);
    auto local_scratch_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_scratch_address);
    local_scratch_ptr[0] = 0;
    WAYPOINT("FMBW");
    do {
        noc_async_read_one_packet(l1_base_noc_addr, local_scratch_address, sizeof(uint32_t), noc);
        noc_async_read_barrier(noc);
        invalidate_l1_cache();
    } while (local_scratch_ptr[0] == 0);
    WAYPOINT("FMBD");
    return local_scratch_ptr[0];
}

}  // namespace tt::tt_fabric
