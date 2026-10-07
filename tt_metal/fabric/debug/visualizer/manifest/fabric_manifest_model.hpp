// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/kernel_types.hpp>

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/struct_layout.hpp"

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
    // A plain read-write register.
    REMOTE_SRC,
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

// The router's tensix mux, which in mux mode feeds the worker channel with the worker's and the siblings' VC0
// traffic.
struct LocalTensixMux {};

// Who writes into a sender channel.
using SenderChannelProducer = std::variant<LocalWorker, LocalTensixMux, SiblingRouterRef>;

// Why a channel's step runs or not.
enum class ChannelStatus : uint32_t {
    // An ERISC runs the channel's step.
    ACTIVE,
    // The kernel does not run the channel's VC on this router (no FABRIC_2D_VC<n>_SERVICED).
    VC_NOT_SERVICED,
    // A VC0 sender other than the worker channel, in mux mode. The tensix mux carries its traffic into the worker
    // channel, and the channel has no buffer slots.
    MUX,
    // The applied channel trimming profile turned the channel off.
    TRIMMED,
};

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

// The NoC and command buffers a receiver forwards on: each packet through the data command buffer, and the credit
// increment that announces it through the sync command buffer.
struct NocForwardConfig {
    tt::tt_metal::NOC noc = tt::tt_metal::NOC::NOC_0;
    NocCmdBuf data_cmd_buf = NocCmdBuf::WR_CMD_BUF;
    NocCmdBuf sync_cmd_buf = NocCmdBuf::WR_CMD_BUF;

    bool operator==(const NocForwardConfig&) const = default;
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

// The group a field belongs to. Mainly for logical grouping of information in decode / visualizer.
enum class FieldCategory : uint8_t {
    LIFECYCLE,
    KERNEL_PARAMS,
    FLOW_CONTROL,
    CONTROL_INFO,
};

// The type at an address or in a stream register.
struct ElementType {
    FieldType type;
    uint32_t size = 0;
};

// Easy accessor to get ElementType from T without manual construction.
template <typename T>
constexpr ElementType element_type() {
    return {field_type<T>(), sizeof(T)};
}

// What a field's argument is, and what reading it needs. The manifest names a field's kind by its type's name in
// snake case (kind_name), so renaming one of these structs renames its kind in the manifest.
namespace kind {

// The address of an `element` in the router's L1.
struct L1Value {
    ElementType element;
};

// A stream id. The field is the stream's `reg` register, which holds an `element`.
struct Stream {
    StreamRegister reg = StreamRegister::REMOTE_SRC;
    ElementType element;
};

struct Number {};

// 0 or 1.
struct Flag {};

// One of an enum's values, which readers name by its schema.
struct Enum {
    ElementType element;
    bool (*is_enumerator)(uint32_t value) = nullptr;
};

}  // namespace kind

using Kind = std::variant<kind::L1Value, kind::Stream, kind::Number, kind::Flag, kind::Enum>;

// A fact the router kernel is fed, read through the field tables (fabric_manifest_fields.hpp).
struct Field {
    std::string_view key;
    // Tag used for logical grouping of information in decode / visualizer.
    FieldCategory category = FieldCategory::LIFECYCLE;
    // Describes what the field's value holds.
    Kind kind;
    // What the kernel is fed: the address, the stream id or the value.
    uint32_t arg = 0;
    // L1Value only: whether the host zeroes the region before launch (get_fabric_router_addresses_to_clear()).
    bool cleared_by_host = false;
};

// The credits a sender channel receives back from the peer's receiver.
struct SenderChannelCredits {
    // Only on VC0 with bubble flow control.
    std::optional<CreditRef> acked;
    CreditRef completed;
};

// A sender channel that takes packets from its producer and sends them over Ethernet to the peer's receiver.
struct SenderChannel {
    ChannelStatus status = ChannelStatus::ACTIVE;
    // IDs of the ERISCs that run the channel's step.
    std::vector<uint32_t> serviced_by;
    // Null when nothing feeds the channel.
    std::optional<SenderChannelProducer> producer;
    L1Region ring_buffer;
    SenderChannelCredits credits;
    // Arguments the kernel is fed for the channel, read through the field table.
    std::vector<Field> fields;
};

// A receiver channel that takes packets from the peer's senders over Ethernet, delivers them locally, and forwards
// them to sibling routers' senders.
struct ReceiverChannel {
    ChannelStatus status = ChannelStatus::ACTIVE;
    // IDs of the ERISCs that run the channel's step.
    std::vector<uint32_t> serviced_by;
    // The VC whose downstream edges the channel's step is given. Null when no ERISC runs the step, or when the step
    // forwards to no sibling.
    std::optional<uint32_t> forwards_on;
    bool forwarding_disabled = false;
    bool intermesh_ingress = false;
    NocForwardConfig forward_noc;
    NocWriteConfig local_write_noc;
    L1Region ring_buffer;
    StreamRef pkts_sent;
    // VC2 only.
    std::optional<StreamRef> free_slots;
};

// A router's channels.
struct Channels {
    // Indexed [vc][channel], over RouterShape's counts.
    std::vector<std::vector<SenderChannel>> senders;
    std::vector<std::vector<ReceiverChannel>> receivers;
};

// A persistent connection on one VC from this router to a sender channel of a router on the same chip and routing
// plane. The receivers forwarding on the VC write through it. In mux mode a VC0 edge into a sibling ends at the
// sibling's tensix mux instead.
struct DownstreamEdge {
    // The kernel's EDGE_<n>: the sibling's compact index among this router's other directions, plus one.
    uint32_t edge = 0;
    SiblingRouterRef target;
    // On the edge's VC.
    uint32_t landing_channel = 0;
    // The landing channel's compact index on the sibling. The builder records it only in 2D.
    std::optional<uint32_t> landing_compact;
    // The NoC core the connection writes to: the sibling's ERISC, or its tensix mux.
    tt::tt_metal::CoreCoord core;
    StreamRef free_slots;
    L1Region teardown_sem;
};

// Information about a router.
struct Router {
    RouterIdentity identity;
    EthLink link;
    RouterShape shape;
    // Always reserved, although not always used. Whether a VC uses them is its mesh's credit_transport backing.
    L1CreditCounters credit_counters;
    Channels channels;
    // Indexed [vc], by edge.
    std::vector<std::vector<DownstreamEdge>> intra_chip_downstream_edges;
    // Router-wide fields passed to the kernel
    std::vector<Field> fields;
    // Per-ERISC fields passed to the kernel, indexed by ERISC id
    std::vector<std::vector<Field>> erisc_fields;
};

// Information about a chip.
struct Chip {
    ZPortRole z_port_role = ZPortRole::NONE;
    std::vector<Router> routers;
};

}  // namespace tt::tt_fabric::manifest
