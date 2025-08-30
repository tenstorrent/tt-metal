// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <cstdint>
#include "tt_metal/lite_fabric/hw/inc/header.hpp"
#include "tt_metal/lite_fabric/hw/inc/constants.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_transaction_id_tracker.hpp"

namespace lite_fabric {

template <template <uint8_t> class ChannelType, auto& BufferSizes, typename Seq>
using ChannelPointersTupleImpl = tt::tt_fabric::ChannelPointersTupleImpl<ChannelType, BufferSizes, Seq>;

template <template <uint8_t> class ChannelType, auto& BufferSizes>
using ChannelPointersTuple = tt::tt_fabric::ChannelPointersTuple<ChannelType, BufferSizes>;

template <uint8_t RECEIVER_NUM_BUFFERS>
using OutboundReceiverChannelPointers = tt::tt_fabric::OutboundReceiverChannelPointers<RECEIVER_NUM_BUFFERS>;
using OutboundReceiverChannelPointersTuple =
    lite_fabric::ChannelPointersTuple<OutboundReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>;
using OutboundReceiverChannelPointersTupleImpl =
    decltype(lite_fabric::ChannelPointersTuple<OutboundReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>::make());

template <uint8_t RECEIVER_NUM_BUFFERS>
using ReceiverChannelPointers = tt::tt_fabric::ReceiverChannelPointers<RECEIVER_NUM_BUFFERS>;
using ReceiverChannelPointersTuple =
    lite_fabric::ChannelPointersTuple<ReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>;
using ReceiverChannelPointersTupleImpl =
    decltype(lite_fabric::ChannelPointersTuple<ReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>::make());

using SenderEthChannelBuffer = tt::tt_fabric::EthChannelBuffer<FabricLiteHeader, SENDER_NUM_BUFFERS_ARRAY[0]>;
using ReceiverEthChannelBuffer = tt::tt_fabric::EthChannelBuffer<FabricLiteHeader, RECEIVER_NUM_BUFFERS_ARRAY[0]>;

using HostInterface = HostToFabricLiteInterface<SENDER_NUM_BUFFERS_ARRAY[0], CHANNEL_BUFFER_SIZE>;

using WriteTridTracker = WriteTransactionIdTracker<
    RECEIVER_NUM_BUFFERS_ARRAY[0],
    NUM_TRANSACTION_IDS,
    0,
    lite_fabric::edm_to_local_chip_noc,
    lite_fabric::edm_to_downstream_noc>;

using RemoteReceiverChannelsType =
    decltype(tt::tt_fabric::EthChannelBuffers<FabricLiteHeader, RECEIVER_NUM_BUFFERS_ARRAY>::make(
        std::make_index_sequence<NUM_RECEIVER_CHANNELS>{}));

using LocalSenderChannelsType =
    decltype(tt::tt_fabric::EthChannelBuffers<FabricLiteHeader, SENDER_NUM_BUFFERS_ARRAY>::make(
        std::make_index_sequence<NUM_SENDER_CHANNELS>{}));

}  // namespace lite_fabric
