
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// this is needed for inclusion of fabric_erisc_datamover_channels.hpp, since we are not
// including 1d_fabric_constants.hpp here, where the constant is originally defined
namespace tt::tt_fabric {
static constexpr uint8_t worker_handshake_noc = 0;
}  // namespace tt::tt_fabric

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/api/tt-metalium/fabric_edm_types.hpp"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"

namespace tt::tt_fabric {

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS>
using FabricMuxChannelBuffer = EthChannelBuffer<FABRIC_MUX_CHANNEL_NUM_BUFFERS>;

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS>
using FabricMuxChannelWorkerInterface = EdmChannelWorkerInterface<FABRIC_MUX_CHANNEL_NUM_BUFFERS>;

using FabricMuxChannelClientLocationInfo = EDMChannelWorkerLocationInfo;

template <uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS>
using WorkerToFabricMuxSender = WorkerToFabricEdmSenderImpl<false, FABRIC_MUX_CHANNEL_NUM_BUFFERS>;

using FabricMuxStatus = EDMStatus;

// Because the mux to producer (worker) (ack) path uses counters, we initialize our EdmChannelWorkerInterface
// (the interface from mux ack to worker) to start at 0. **Note** that if this is ever updated to be free slots
// based instead, we'd initialize to num buffer slots (NUM_EDM_BUFFERS) instead.
constexpr size_t MUX_TO_WORKER_INTERFACE_STARTING_READ_COUNTER_VALUE = 0;

}  // namespace tt::tt_fabric
