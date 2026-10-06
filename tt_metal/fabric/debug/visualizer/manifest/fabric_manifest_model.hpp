// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/kernel_types.hpp>

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"

// The fabric manifest's model. The collector fills in what the router builders know, and the writer
// (write_fabric_manifest) adds what only ControlPlane and the cluster know (routing plane, peer, cross-host,
// wrap, cores) and serializes the manifest.
namespace tt::tt_fabric::manifest {

// A capturable range of L1.
struct L1Region {
    uint32_t address = 0;
    uint32_t size = 0;
    // Set for arrays: size == num_elements * size_per_element.
    std::optional<uint32_t> num_elements;
    std::optional<uint32_t> size_per_element;
    // Schema string, e.g. "u32" or "struct:EDMChannelWorkerLocationInfo" (schema_name()).
    std::string schema;
    // If true, the host zeroes the region before launch (get_fabric_router_addresses_to_clear()).
    bool cleared_by_host = false;
};

// Which register of a stream a stream ref reads.
enum class StreamRegister : uint8_t {
    // The increment-on-write credit or slot count.
    BUF_SPACE_AVAILABLE,
};

// A stream register reference.
struct StreamRef {
    uint32_t stream_id = 0;
    StreamRegister reg = StreamRegister::BUF_SPACE_AVAILABLE;
    std::string schema;
};

// One of a router's L1 credit counter arrays (L1CreditCounters).
enum class CreditCounterArray : uint8_t {
    TO_SENDER_ACK,
    TO_SENDER_COMPLETION,
    RECEIVER_ACK,
    RECEIVER_COMPLETION,
};

// One element of a credit counter array on the same router.
struct CounterRef {
    CreditCounterArray array = CreditCounterArray::TO_SENDER_ACK;
    uint32_t index = 0;
};

// Credits are held in a stream register or in an element of an L1 counter array.
using CreditRef = std::variant<StreamRef, CounterRef>;

// A router on the same chip and routing plane, named by the direction it faces.
struct SiblingRouterRef {
    eth_chan_directions direction = eth_chan_directions::EAST;
};

// The chip's local worker, as the producer of a sender channel.
struct LocalWorker {};

// Who writes into a sender channel.
using SenderChannelProducer = std::variant<LocalWorker, SiblingRouterRef>;

// A NoC command buffer.
enum class NocCmdBuf : uint32_t {
    WR_CMD_BUF = FabricEriscDatamoverConfig::WR_CMD_BUF,
    RD_CMD_BUF = FabricEriscDatamoverConfig::RD_CMD_BUF,
    WR_REG_CMD_BUF = FabricEriscDatamoverConfig::WR_REG_CMD_BUF,
    AT_CMD_BUF = FabricEriscDatamoverConfig::AT_CMD_BUF,
};

// The NoC and command buffer a write goes out on.
struct NocWriteConfig {
    tt::tt_metal::NOC noc = tt::tt_metal::NOC::NOC_0;
    NocCmdBuf cmd_buf = NocCmdBuf::WR_CMD_BUF;
};

// A router's L1 credit counter arrays, shared by every VC that uses counter credits. The to_sender arrays are
// indexed by this router's sender compact index, and the receiver arrays by the peer router's sender compact index
// (the receiver counts credits for the peer's sender channels).
struct L1CreditCounters {
    L1Region to_sender_ack;
    L1Region to_sender_completion;
    L1Region receiver_ack;
    L1Region receiver_completion;
};

// High level information about a particular router.
struct RouterIdentity {
    uint32_t eth_chan = 0;
};

// Ethernet link information about a particular router.
struct EthLink {
    eth_chan_directions direction = eth_chan_directions::EAST;
    EdgeCapability edge_capability = EdgeCapability::INTRAMESH_CARDINAL;
    bool is_dispatch_link = false;
};

// Information about the "shape" of a router, i.e. the number of VCs, senders, receivers, and active ERISCs.
struct RouterShape {
    uint32_t num_vcs = 0;
    std::array<uint32_t, builder_config::MAX_NUM_VCS> senders_per_vc = {};
    std::array<uint32_t, builder_config::MAX_NUM_VCS> receivers_per_vc = {};
    uint32_t num_active_eriscs = 0;
    bool channel_trimming_overrides_applied = false;
    bool vc0_bubble_flow_control = false;
};

// The credits a sender channel receives back from the peer's receiver.
struct SenderChannelCredits {
    // Only on VC0 with bubble flow control.
    std::optional<CreditRef> acked;
    CreditRef completed;
};

// The L1 a sender channel's producer uses to connect and to report where it writes.
struct SenderChannelControlInfo {
    L1Region connection;
    L1Region conn_info;
    L1Region buffer_index_sem;
};

// A sender channel that takes packets from its producer and sends them over Ethernet to the peer's receiver.
struct SenderChannel {
    // IDs of the ERISCs that run the channel's step.
    std::vector<uint32_t> serviced_by;
    // Null when nothing feeds the channel.
    std::optional<SenderChannelProducer> producer;
    bool is_injection_channel = false;
    NocWriteConfig producer_credit_return;
    L1Region ring_buffer;
    StreamRef free_slots;
    SenderChannelCredits credits;
    SenderChannelControlInfo control_info;
};

// A router's channels.
struct Channels {
    // Indexed [vc][channel], over RouterShape's counts.
    std::vector<std::vector<SenderChannel>> senders;
};

// Information about a router.
struct Router {
    RouterIdentity identity;
    EthLink link;
    RouterShape shape;
    // Always reserved, although not always used. Whether a VC uses them is its mesh's credit_transport backing.
    L1CreditCounters credit_counters;
    Channels channels;
};

// Information about a chip.
struct Chip {
    ZPortRole z_port_role = ZPortRole::NONE;
    std::vector<Router> routers;
};

}  // namespace tt::tt_fabric::manifest
