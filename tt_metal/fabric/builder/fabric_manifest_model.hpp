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
#include <tt-metalium/experimental/fabric/fabric_telemetry.hpp>
#include <tt-metalium/kernel_types.hpp>

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"

// The fabric manifest's router model. The collector fills in what the router builders know and the
// writer (serialize_fabric_manifest_to_file) adds what only ControlPlane knows
// (routing plane, router ids, peer, cross-host, wrap, cores) and serializes the manifest.
namespace tt::tt_fabric::manifest {

// A capturable range of L1.
struct L1Region {
    uint32_t address = 0;
    uint32_t size = 0;
    // Set for arrays: size == num_elements * size_per_element.
    std::optional<uint32_t> num_elements;
    std::optional<uint32_t> size_per_element;
    // Prefixed schema string, e.g. "u32", "enum:EDMStatus", "struct:handshake_info_t".
    std::string schema;
    // If true, the host zeroes the region before launch (get_fabric_router_addresses_to_clear()).
    bool host_cleared = false;
};

// An uncaptured span of L1. In an effort to include all L1 regions, this serves
// as a catch-all for any L1 region that is not captured by a more specific type.
struct L1Span {
    uint32_t address = 0;
    uint32_t size = 0;
};

// Stream register types.
enum class StreamRegister : uint8_t {
    // Credit or slot count
    BUF_SPACE_AVAILABLE,
    // Stream 30 and 31 read REMOTE_SRC and name their state enum.
    REMOTE_SRC
};

// Stream register reference.
struct StreamRef {
    uint32_t stream_id = 0;
    StreamRegister reg = StreamRegister::BUF_SPACE_AVAILABLE;
    std::string schema = "u32";
};

// One element of an L1 array on the same router.
struct ArrayRef {
    // Array path, relative to the router, in the manifest JSON, e.g. "credit_counters/to_sender_ack"
    std::string array;
    // Element index in the array
    uint32_t index = 0;
};

// Credit reference. Credits can be stored in a stream register or an l1 counter, where l1 counters
// are represented as an array element.
using CreditRef = std::variant<StreamRef, ArrayRef>;

// A router on the same chip and routing plane, named by the direction it faces.
struct SiblingRouterRef {
    eth_chan_directions direction = eth_chan_directions::EAST;
};

// Used to represent the local worker as a producer of a sender channel.
struct LocalWorker {};

// Who writes into a sender channel.
using SenderChannelProducer = std::variant<LocalWorker, SiblingRouterRef>;

// NoC command buffers type (mirrors FabricEriscDatamoverConfig's constants).
// TODO: replace those constants with an enum in the builder and use it here.
enum class NocCmdBuf : uint8_t {
    WR_CMD_BUF = 0,      // large writes
    RD_CMD_BUF = 1,      // reads
    WR_REG_CMD_BUF = 2,  // small writes (registers, semaphores)
    AT_CMD_BUF = 3,      // atomics
};

// NoC write resources configuration.
struct NocWriteConfig {
    tt::tt_metal::NOC noc = tt::tt_metal::NOC::NOC_0;
    NocCmdBuf cmd_buf = NocCmdBuf::WR_CMD_BUF;
};

// NoC forward resources configuration. Packets go through the data command buffer, and the credit
// increment that announces each one through the sync command buffer, both on the same NoC.
struct NocForwardConfig {
    tt::tt_metal::NOC noc = tt::tt_metal::NOC::NOC_0;
    NocCmdBuf data_cmd_buf = NocCmdBuf::WR_CMD_BUF;
    NocCmdBuf sync_cmd_buf = NocCmdBuf::WR_CMD_BUF;
};

// High level information about a particular router.
struct RouterIdentity {
    uint32_t eth_chan = 0;
};

// Ethernet link information about a particular router.
struct EthLink {
    eth_chan_directions direction = eth_chan_directions::EAST;
    EdgeCapability edge_capability = EdgeCapability::INTRAMESH_CARDINAL;
    bool dispatch_link = false;
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

// A router's L1 credit counters, shared by every VC that uses counter credits. The to_sender arrays are
// indexed by this router's sender compact index and the receiver arrays by the peer's. The writer emits
// that as each array's index_space.
struct L1CreditCounters {
    L1Region to_sender_ack;
    L1Region to_sender_completion;
    L1Region receiver_ack;
    L1Region receiver_completion;
};

// Credit management info for a sender channel.
struct SenderChannelCredits {
    // Only with VC0 bubble flow control.
    std::optional<CreditRef> acked;
    CreditRef completed;
};

// Control information for a sender channel.
struct SenderChannelControlInfo {
    L1Region connection;
    L1Region conn_info;
    // Only on worker-fed (non-persistent) channels.
    std::optional<L1Region> buffer_index_sem;
};

// Information about a sender channel.
struct SenderChannel {
    // IDs of the ERISCs that service the channel.
    std::vector<uint32_t> serviced_by;
    // Null when nothing feeds the channel (no upstream producer connected to it).
    std::optional<SenderChannelProducer> producer;
    bool is_injection_channel = false;
    NocWriteConfig producer_credit_return;
    L1Region ring_buffer;
    StreamRef free_slots;
    SenderChannelCredits credits;
    SenderChannelControlInfo control_info;
};

// Information about a downstream edge for a receiver channel.
struct DownstreamEdge {
    // The kernel's 1-based compact slot index (EDGE_1..EDGE_4).
    uint32_t edge = 0;
    SiblingRouterRef target;
    uint32_t landing_vc = 0;
    uint32_t landing_channel = 0;
    StreamRef free_slots;
    L1Region teardown_sem;
};

// Information about a receiver channel.
struct ReceiverChannel {
    // IDs of the ERISCs that service the channel.
    std::vector<uint32_t> serviced_by;
    bool forwarding_disabled = false;
    bool intermesh_ingress = false;
    NocForwardConfig forward_noc;
    NocWriteConfig local_write_noc;
    L1Region ring_buffer;
    StreamRef pkts_sent;
    // VC2 only.
    std::optional<StreamRef> free_slots;
    std::vector<DownstreamEdge> downstream_edges;
};

// Information about all channels on a router.
struct Channels {
    // Indexed [vc][channel], over RouterShape's counts.
    std::vector<std::vector<SenderChannel>> senders;
    std::vector<std::vector<ReceiverChannel>> receivers;
    // Blackhole only (category 9). Written under channels.senders.
    std::optional<L1Region> notify_worker_src;
};

// Roles of a handshake.
enum class HandshakeRole : uint8_t {
    SENDER,
    RECEIVER
};

// Information about a handshake.
struct Handshake {
    L1Region region;
    HandshakeRole role = HandshakeRole::SENDER;
};

// Lifecycle features an ERISC runs.
struct EriscFeatures {
    bool handshake_enabled = false;
    bool context_switch_enabled = false;
    bool interrupts_enabled = false;
    uint32_t teardown_check_iterations = 0;
    // The FabricTelemetryStatistic bits this ERISC collects; 0 when its telemetry is off.
    FabricTelemetryStatisticMask telemetry_stats_mask = 0;
};

// How the router's ERISCs context switch.
struct ContextSwitchPolicy {
    FabricEriscDatamoverContextSwitchType mode = FabricEriscDatamoverContextSwitchType::WAIT_FOR_IDLE;
    uint32_t interval = 0;
};

// Behavior of a spin wait for a tx queue.
struct TxqSpinWait {
    bool send_data = false;
    bool completion_ack = false;
};

// Router kernel parameters, shared by all of its ERISCs.
struct RouterKernelParams {
    bool wait_for_host_signal = false;
    ContextSwitchPolicy context_switch;
    uint32_t handshake_context_switch_timeout = 0;
    TxqSpinWait txq_spin_wait;
    uint32_t txq_accept_ahead = 0;
    bool risc_cpu_data_cache = false;
};

// Information about the lifecycle of a router.
struct Lifecycle {
    L1Region edm_status;
    L1Region termination_signal;
    L1Region local_sync;
    L1Region local_tensix_sync;
    Handshake handshake;
    // Two ERISCs only.
    std::optional<StreamRef> erisc_sync;
    // Two ERISCs only; the ERISC with context switch enabled leads it.
    std::optional<StreamRef> retrain_sync;
    // Indexed by ERISC id, one per active ERISC.
    std::vector<EriscFeatures> erisc_features;
    RouterKernelParams kernel_params;
};

// Each buffer is present when the builder allocated it. Blackhole always reserves perf telemetry, even when
// it is off.
struct Diagnostics {
    std::optional<L1Region> perf_telemetry;
    std::optional<L1Region> code_profiling;
    std::optional<L1Region> channel_trimming;
};

// Information about a router.
struct Router {
    RouterIdentity identity;
    EthLink link;
    RouterShape shape;
    // Always reserved, whether a VC uses them is its credit_transport backing.
    L1CreditCounters credit_counters;
    Channels channels;
    Lifecycle lifecycle;
    Diagnostics diagnostics;
    // From the end of the last ring to max_l1_loading_size.
    // TODO: end at the debug block base addr instead when debug instrumentation is on.
    L1Span leftover_l1;
};

// Information about a chip.
struct Chip {
    ZPortRole z_port_role = ZPortRole::NONE;
    // The writer turns this into the chip's local_sync.master path.
    uint32_t master_router_chan = 0;
    std::vector<Router> routers;
};

}  // namespace tt::tt_fabric::manifest
