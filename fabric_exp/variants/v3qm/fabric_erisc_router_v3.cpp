// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric ERISC router (v3): moves fabric packets across one Ethernet link.
//
// Channels. A sender channel takes packets from a worker (channel 0 of a VC) or from an upstream router on this chip
// (a forwarding channel) and sends them over the link into the far router's receiver channel of the same VC. A
// receiver channel takes packets off the link and writes each one to this chip (local delivery), forwards it to the
// next router's sender channel on this chip, or both. Credits flow back the same way: the far receiver returns a
// completion per slot once no NoC command still reads the slot, and the sender channel then returns the slot to its
// upstream (the worker, or the upstream router's free-slots register).
//
// Programs. MY_ERISC_ID and the compile-time args select one of three programs:
//   - sender_erisc (fabric_erisc_router_v3_sender_erisc.hpp): ERISC 0 of a two-ERISC Blackhole router, which services
//     only sender channels.
//   - receiver_erisc (fabric_erisc_router_v3_receiver_erisc.hpp): ERISC 1 of a two-ERISC Blackhole router, which
//     services only receiver channels.
//   - run_fabric_edm_main_loop, "the generic loop": every other build (single-ERISC mode, Wormhole, the speedy path,
//     VC2, UDM, deadlock avoidance on ERISC 1, ...). Each specialized program lists the builds it covers in its
//     `runs` constant; it produces the same packets, credits and completions as the generic loop.
// kernel_main initializes the router and runs sender_erisc or the generic loop; receiver_erisc::run_kernel is ERISC 1's
// whole kernel when receiver_erisc::runs. In two-ERISC mode only the ERISC that services the receiver channels sets
// up the receiver side, and both ERISCs pass the same wait_for_other_local_erisc() syncs in the same order (six in
// init, three in teardown): the two init paths must change together.
//
// FABRIC_ERISC_ROUTER_V3 selects this router's policies in shared headers (receiver completion batching in
// fabric_erisc_router_ct_args.hpp, deferred flushed atomics in fabric_edm_packet_transmission.hpp), so it is defined
// before the first of them is included.
#define FABRIC_ERISC_ROUTER_V3 1

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "internal/ethernet/tunneling.h"

#include "hostdev/fabric_edm_packet_header.hpp"
#include <tt-metalium/experimental/fabric/edm_fabric_counters.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"
#include "hostdevcommon/fabric_common.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_eth_handshake.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_adapter.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_header_validate.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_transaction_id_tracker.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_utils.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_tmp_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_packet_recorder.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_bandwidth_telemetry.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_code_profiling.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_channel_traits.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/router_data_cache.hpp"

#include "noc_overlay_parameters.h"
#include "api/alignment.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_txq_setup.h"
#include "hostdevcommon/fabric_telemetry_msgs.h"
#ifdef FABRIC_2D
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#endif

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

using namespace tt::tt_fabric;
// Type alias for the 1D low-latency hop fields.
using LowLatencyFields = tt::tt_fabric::RoutingFieldsConstants::LowLatency;

// read and write stream scratch register store values as uint32_t
enum class CoordinatedEriscContextSwitchState : uint32_t {
    // Initially set by the master (erisc0) in entrance of kernel_main() and is the default state. erisc1 polls for this
    // state at the end of the handshake
    NORMAL_EXECUTION = 0,
    // Set by master to signal intent on beginning handshake. Checked by erisc1 before it begins handshake
    RETRAIN_INTENT = 1,
    // Set by erisc1 to indicate it's cooperation. Polled by master before it runs the retrain to ensure that it has
    // erisc1's cooperation
    INTENT_ACK = 2,
    // Set by erisc0 to indicate completion of retrain. Polled by erisc1
    RETRAIN_COMPLETE = 3,
    // Set by erisc1 to indicate it has seen the completion of retrain. Polled by erisc0 before it sets register back to
    // NORMAL_EXECUTION
    COMPLETE_ACK = 4,
};

using CoordinatedEriscCtxType = std::underlying_type_t<CoordinatedEriscContextSwitchState>;

template <typename HEADER_TYPE, uint8_t NUM_BUFFERS>
using SenderEthChannel = StaticSizedSenderEthChannel<HEADER_TYPE, NUM_BUFFERS>;

static constexpr bool PERF_TELEMETRY_DISABLED = perf_telemetry_mode == PerfTelemetryRecorderType::NONE;
static constexpr bool PERF_TELEMETRY_LOW_RESOLUTION_BANDWIDTH =
    perf_telemetry_mode == PerfTelemetryRecorderType::LOW_RESOLUTION_BANDWIDTH;
using PerfTelemetryRecorder = std::conditional_t<
    PERF_TELEMETRY_LOW_RESOLUTION_BANDWIDTH,
    LowResolutionBandwidthTelemetry,
    std::conditional_t<PERF_TELEMETRY_DISABLED, bool, std::nullptr_t>>;

template <uint8_t SENDER_NUM_BUFFERS>
using RouterToRouterSender = tt::tt_fabric::EdmToEdmSender<SENDER_NUM_BUFFERS>;

static constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> sender_channel_free_slots_stream_ids = {
    sender_channel_0_free_slots_stream_id,
    sender_channel_1_free_slots_stream_id,
    sender_channel_2_free_slots_stream_id,
    sender_channel_3_free_slots_stream_id,
    sender_channel_4_free_slots_stream_id,
    sender_channel_5_free_slots_stream_id,
    sender_channel_6_free_slots_stream_id,
    sender_channel_7_free_slots_stream_id,
    sender_channel_8_free_slots_stream_id,
    sender_channel_9_free_slots_stream_id};

// Each VC's sender free-slots registers form one contiguous group. Boundaries come from the VC-start
// CT args rather than pinned literals, so a future repack still holds.
constexpr bool is_contiguous_stream_span(const uint32_t* ids, size_t count) {
    for (size_t i = 1; i < count; ++i) {
        if (ids[i] != ids[i - 1] + 1) {
            return false;
        }
    }
    return true;
}
static_assert(is_contiguous_stream_span(sender_channel_free_slots_stream_ids.data(), VC1_FABRIC_POSITION_START));
static_assert(is_contiguous_stream_span(
    sender_channel_free_slots_stream_ids.data() + VC1_FABRIC_POSITION_START,
    VC2_FABRIC_POSITION_START - VC1_FABRIC_POSITION_START));
static_assert(
    sender_channel_free_slots_stream_ids[VC2_FABRIC_POSITION_START] ==
    (ACTUAL_VC2_SENDER_CHANNELS > 0 ? tt::tt_fabric::connection_interface::vc2_sender_free_slots_stream_id
                                    : tt::tt_fabric::k_unused_stream_id));

// The sentinel marks an inactive consumer, e.g. a VC whose credits travel through L1 counters and
// reads no register at all. It is out of register range, so an inactive entry can clobber nothing.
template <uint32_t STREAM_ID>
FORCE_INLINE void init_stream_reg_if_used(uint32_t value) {
    if constexpr (STREAM_ID != tt::tt_fabric::k_unused_stream_id) {
        init_ptr_val<STREAM_ID>(value);
    }
}

// For 2D fabric: maps compact index to downstream direction for each my_direction
// For 1D fabric: only 1 downstream direction per router (EAST forwards to WEST in 1D linear topology)
constexpr uint32_t MAX_DOWNSTREAM_EDM_COUNT = 4;
#if defined(FABRIC_2D)
constexpr uint32_t edm_index_to_edm_direction[eth_chan_directions::COUNT][MAX_DOWNSTREAM_EDM_COUNT] = {
    {eth_chan_directions::WEST,
     eth_chan_directions::NORTH,
     eth_chan_directions::SOUTH,
     eth_chan_directions::Z},  // EAST router
    {eth_chan_directions::EAST,
     eth_chan_directions::NORTH,
     eth_chan_directions::SOUTH,
     eth_chan_directions::Z},  // WEST router
    {eth_chan_directions::EAST,
     eth_chan_directions::WEST,
     eth_chan_directions::SOUTH,
     eth_chan_directions::Z},  // NORTH router
    {eth_chan_directions::EAST,
     eth_chan_directions::WEST,
     eth_chan_directions::NORTH,
     eth_chan_directions::Z},  // SOUTH router
    {eth_chan_directions::EAST,
     eth_chan_directions::WEST,
     eth_chan_directions::NORTH,
     eth_chan_directions::SOUTH},  // Z router
};

constexpr uint32_t get_vc0_downstream_sender_channel_free_slots_stream_id(uint32_t compact_index) {
    auto ds_edm_direction = edm_index_to_edm_direction[my_direction][compact_index];
    if (my_direction > ds_edm_direction) {
        // downstream sender channel = my_direction
        // stream id = sender_channel_free_slots_stream_ids[downstream sender channel]
        return sender_channel_free_slots_stream_ids[my_direction];
    } else {
        // downstream sender channel = my_direction + 1
        // stream id = sender_channel_free_slots_stream_ids[downstream sender channel]
        return sender_channel_free_slots_stream_ids[(1 + my_direction)];
    }
}

// VC1 downstream sender channel mapping (for inter-mesh routing)
// Same direction rules as VC0, offset by VC1's fabric base to skip VC0's channels. The base is a CT
// arg because VC0's width varies: 4 on legacy fabrics, 5 on one with a boundary router.
constexpr uint32_t get_vc1_downstream_sender_channel_free_slots_stream_id(uint32_t compact_index) {
    auto ds_edm_direction = edm_index_to_edm_direction[my_direction][compact_index];
    if (my_direction > ds_edm_direction) {
        // downstream sender channel = (VC1 fabric base - 1) + my_direction
        // stream id = sender_channel_free_slots_stream_ids[downstream sender channel]
        return sender_channel_free_slots_stream_ids[VC1_FABRIC_POSITION_START - 1 + my_direction];
    } else {
        // downstream sender channel = (VC1 fabric base) + my_direction
        // stream id = sender_channel_free_slots_stream_ids[downstream sender channel]
        return sender_channel_free_slots_stream_ids[VC1_FABRIC_POSITION_START + my_direction];
    }
}
#endif

FORCE_INLINE constexpr eth_chan_directions map_compact_index_to_direction(size_t compact_index) {
#if defined(FABRIC_2D)
    return static_cast<eth_chan_directions>(edm_index_to_edm_direction[my_direction][compact_index]);
#else
    return static_cast<eth_chan_directions>(compact_index);
#endif
}

static constexpr std::array<uint32_t, NUM_ROUTER_CARDINAL_DIRECTIONS> vc_0_free_slots_stream_ids = {
    vc_0_free_slots_from_downstream_edge_1_stream_id,
    vc_0_free_slots_from_downstream_edge_2_stream_id,
    vc_0_free_slots_from_downstream_edge_3_stream_id,
    vc_0_free_slots_from_downstream_edge_4_stream_id};

static constexpr std::array<uint32_t, NUM_ROUTER_CARDINAL_DIRECTIONS> vc_1_free_slots_stream_ids = {
    vc_1_free_slots_from_downstream_edge_1_stream_id,
    vc_1_free_slots_from_downstream_edge_2_stream_id,
    vc_1_free_slots_from_downstream_edge_3_stream_id,
    vc_1_free_slots_from_downstream_edge_4_stream_id};

template <size_t Compact>
constexpr uint32_t vc0_local_free_slots_stream_id() {
    return vc_0_free_slots_stream_ids[Compact];
}
constexpr uint32_t vc0_local_free_slots_stream_id(size_t compact) { return vc_0_free_slots_stream_ids[compact]; }

template <size_t Compact>
constexpr uint32_t vc1_local_free_slots_stream_id() {
    return vc_1_free_slots_stream_ids[Compact];
}
constexpr uint32_t vc1_local_free_slots_stream_id(size_t compact) { return vc_1_free_slots_stream_ids[compact]; }

// tracks if the main loop made any progress. If many loop iterations were completed without
// did_something=true (i.e. no progress was made), then we allow for context switch in case
// the link is down
bool did_something;

// The Ethernet firmware's heartbeat word (eth_status_t.heartbeat[0], MEM_SYSENG_ETH_HEARTBEAT on Blackhole). While the
// router runs in place of the base firmware, each main loop keeps the word moving every 64 outer iterations, with
// UMD's fabric signature (0xDCBA) in the upper half.
#if defined(ARCH_BLACKHOLE)
constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_ADDR = 0x7CC70;
#else
constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_ADDR = 0x1F80;
#endif

FORCE_INLINE void record_sender_channel_usage(size_t sender_channel_index, size_t packet_size_bytes) {
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        channel_trimming_usage_recorder.set_sender_channel_used(sender_channel_index);
        channel_trimming_usage_recorder.update_sender_channel_packet_size(
            sender_channel_index, static_cast<uint16_t>(packet_size_bytes));
    }
}

// A flushed atomic (an atomic that must follow every earlier write to the local chip) may be deferred past later
// packets instead of the receiver spinning until those writes are acknowledged. Needs the batched-completion receiver
// (one transaction ID per slot, 16 slots or more) alone on its ERISC and one NoC for local and downstream writes, so
// that one global check covers every outstanding write.
//
// Deferral protocol (execute_local_semaphore_inc, drain_pending_atomic_flush; state `pending_atomic_flush`):
//   - The deferred atomic is staged in write_at_cmd_buf (address, data, control) and issued later from there without
//     re-checking the buffer. Until it is issued or drained, no other NoC command may use write_at_cmd_buf: packets
//     that would (atomics, inline writes, stateless forwards, whose credit write is noc_inline_dw_write) drain it
//     first. Only plain local writes and stateful forwards may bypass it.
//   - Encoding: 0 none. 1 strict: issued once every write on the NoC is acknowledged, at the latest before the next
//     packet's NoC commands. 0x10 | s relaxed, s = the trid of the last write before the atomic: at the first later
//     write (on another trid) the receiver checks that every outstanding write is on trid s or s - 1 and records the
//     one or two still outstanding as 0x100 | b << 4 | a; the atomic is issued once those are flushed. Transaction IDs
//     are 4 bits here (NUM_TRANSACTION_IDS == 16, one per slot, assigned in slot order).
static_assert(RX_CH_TRID_STARTS[0] == 0, "the deferral's trid arithmetic (& 0xF) assumes channel 0's trids start at 0");
template <bool enable_first_level_ack>
constexpr bool can_defer_local_atomic_flush() {
#if defined(ARCH_BLACKHOLE) && defined(FABRIC_2D) && !defined(FABRIC_2D_VC1_ACTIVE) && !defined(FABRIC_2D_VC2_SERVICED)
    return !enable_deadlock_avoidance && !enable_first_level_ack && !udm_mode && !any_sender_channel_serviced &&
           fuse_receiver_flush_and_completion_ptr && tt::tt_fabric::local_chip_noc_equals_downstream_noc &&
           tt::tt_fabric::edm_to_local_chip_noc == noc_index && NUM_RECEIVER_CHANNELS == 1 &&
           RECEIVER_NUM_BUFFERS_ARRAY[0] >= NUM_TRANSACTION_IDS && NUM_TRANSACTION_IDS == 16;
#else
    return false;
#endif
}

FORCE_INLINE void generic_rx_local_delivery(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
    [[maybe_unused]] uint16_t payload_size_bytes,
    uint8_t transaction_id,
    uint8_t rx_channel_id) {
    auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_start);
    execute_chip_unicast_to_local_chip_impl(
        packet_start, packed.payload_size_bytes, packed.noc_send_type, transaction_id, rx_channel_id);
}

template <typename DownstreamSenderT>
FORCE_INLINE bool can_forward_packet_completely(
    ROUTING_FIELDS_TYPE cached_routing_fields, DownstreamSenderT& downstream_edm_interface) {
    // We always check if it is the terminal mcast packet value. We can do this because all unicast packets have the
    // mcast terminal value masked in to the routing field. This simplifies the check here to a single compare.
    bool deliver_locally_only;
    if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::RoutingFields>) {
        deliver_locally_only = cached_routing_fields.value == tt::tt_fabric::RoutingFields::LAST_MCAST_VAL;
    } else if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::LowLatencyRoutingFields>) {
        deliver_locally_only =
            (cached_routing_fields.value & LowLatencyFields::FIELD_MASK) == LowLatencyFields::WRITE_ONLY;
    }
    return deliver_locally_only ||
           downstream_edm_interface.template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>();
}

// Map downstream direction to compact array index [0-3], excluding my_direction.
// Routers never forward back out their own facing, so that slot is unused and the remaining four
// directions pack densely.
// - EAST router  (my_direction=0): WEST(1)->0, NORTH(2)->1, SOUTH(3)->2, Z(4)->3
// - WEST router  (my_direction=1): EAST(0)->0, NORTH(2)->1, SOUTH(3)->2, Z(4)->3
// - NORTH router (my_direction=2): EAST(0)->0, WEST(1)->1,  SOUTH(3)->2, Z(4)->3
// - SOUTH router (my_direction=3): EAST(0)->0, WEST(1)->1,  NORTH(2)->2, Z(4)->3
// - Z router     (my_direction=4): EAST(0)->0, WEST(1)->1,  NORTH(2)->2, SOUTH(3)->3
constexpr size_t direction_to_compact_index_map[eth_chan_directions::COUNT][eth_chan_directions::COUNT] = {
    {0, 0, 1, 2, 3},  // EAST router -> WEST, NORTH, SOUTH, Z
    {0, 0, 1, 2, 3},  // WEST router -> EAST, NORTH, SOUTH, Z
    {0, 1, 0, 2, 3},  // NORTH router -> EAST, WEST, SOUTH, Z
    {0, 1, 2, 0, 3},  // SOUTH router -> EAST, WEST, NORTH, Z
    {0, 1, 2, 3, 0},  // Z router -> EAST, WEST, NORTH, SOUTH
};

template <eth_chan_directions downstream_direction>
FORCE_INLINE constexpr size_t map_downstream_direction_to_compact_index() {
    return direction_to_compact_index_map[my_direction][downstream_direction];
}

FORCE_INLINE constexpr size_t map_downstream_direction_to_compact_index(eth_chan_directions downstream_direction) {
    return direction_to_compact_index_map[my_direction][downstream_direction];
}

template <eth_chan_directions downstream_direction>
FORCE_INLINE constexpr size_t get_downstream_edm_interface_index() {
    return map_downstream_direction_to_compact_index<downstream_direction>();
}

FORCE_INLINE constexpr size_t get_downstream_edm_interface_index(eth_chan_directions downstream_direction) {
    // Map downstream direction to compact array index (excluding router's own direction)
    return map_downstream_direction_to_compact_index(downstream_direction);
}

// Whether this router actually instantiates the downstream slot a direction maps to. A compact index
// is fixed by the direction pair alone, so a narrow router simply leaves the higher slots absent --
// X_RING_ONLY gives an E/W-facing router only its opposite, so slots 1 and 2 do not exist there.
// NOTE (Z-arm invariant): a router with no Z downstream must never index past its downstream array.
// fwd_dirs<MY_DIR>() always lists Z, but the linear LIVE scan only emits a runtime ACTION_Z test when
// that slot's compact index is < DOWNSTREAM_EDM_SIZE. Wormhole has no Z sender, so the test is
// if-constexpr gone. num_z_ports is not visible to the kernel to cross-check against.
template <size_t DOWNSTREAM_EDM_SIZE, eth_chan_directions DIRECTION>
constexpr bool dispatch_arm_is_realizable() {
    return get_downstream_edm_interface_index<DIRECTION>() < DOWNSTREAM_EDM_SIZE;
}

// The selected downstream array determines the carrier VC. Query its adapter so crossover traffic
// reads VC1 credits rather than the compact-indexed VC0 counter.
template <typename DownstreamSenderT, eth_chan_directions DIRECTION, size_t DOWNSTREAM_EDM_SIZE>
FORCE_INLINE bool check_downstream_has_space(
    [[maybe_unused]] std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces) {
    if constexpr (DIRECTION == my_direction) {
        return true;
    } else {
        constexpr auto edm_index = get_downstream_edm_interface_index<DIRECTION>();
        return downstream_edm_interfaces[edm_index].template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>();
    }
}

template <
    typename DownstreamSenderT,
    typename LocalRelayInterfaceT,
    eth_chan_directions DIRECTION,
    size_t DOWNSTREAM_EDM_SIZE>
FORCE_INLINE bool check_downstream_has_space(
    [[maybe_unused]] std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    LocalRelayInterfaceT& local_relay_interface) {
    if constexpr (DIRECTION == my_direction) {
        if constexpr (udm_mode) {
            return local_relay_interface.template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>();
        } else {
            return true;
        }
    } else {
        constexpr auto edm_index = get_downstream_edm_interface_index<DIRECTION>();
        return downstream_edm_interfaces[edm_index].template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>();
    }
}

template <
    typename DownstreamSenderT,
    typename LocalRelayInterfaceT,
    size_t DOWNSTREAM_EDM_SIZE,
    eth_chan_directions... DIRECTIONS>
FORCE_INLINE bool downstreams_have_space(
    std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    LocalRelayInterfaceT& local_relay_interface) {
    return (
        ... && check_downstream_has_space<DownstreamSenderT, LocalRelayInterfaceT, DIRECTIONS, DOWNSTREAM_EDM_SIZE>(
                   downstream_edm_interfaces, local_relay_interface));
}

#ifndef FABRIC_2D
// !!!WARNING!!! - MAKE SURE CONSUMER HAS SPACE BEFORE CALLING
template <uint8_t rx_channel_id, typename DownstreamSenderT>
FORCE_INLINE void receiver_forward_packet(
    tt_l1_ptr PACKET_HEADER_TYPE* packet_start,
    ROUTING_FIELDS_TYPE cached_routing_fields,
    DownstreamSenderT& downstream_edm_interface,
    uint8_t transaction_id) {
    // disable when dprint enabled due to noc cmd buf usage of DPRINT
    constexpr bool ENABLE_STATEFUL_NOC_APIS =
#if !defined(DEBUG_PRINT_ENABLED) and !defined(WATCHER_ENABLED)
        !FORCE_ALL_PATHS_TO_USE_SAME_NOC && true;
#else
        false;
#endif
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::RoutingFields>) {
        // If the packet is a terminal packet, then we can just deliver it locally
        bool start_distance_is_terminal_value =
            (cached_routing_fields.value & tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK) ==
            tt::tt_fabric::RoutingFields::LAST_HOP_DISTANCE_VAL;
        uint16_t payload_size_bytes = packet_start->payload_size_bytes;
        bool not_last_destination_device = cached_routing_fields.value != tt::tt_fabric::RoutingFields::LAST_MCAST_VAL;
        if (not_last_destination_device) {
            forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                packet_start, payload_size_bytes, cached_routing_fields, downstream_edm_interface, transaction_id);
            channel_trimming_usage_recorder.set_sender_channel_forwarded_to(rx_channel_id, 1);
        }
        if (start_distance_is_terminal_value) {
            generic_rx_local_delivery(packet_start, payload_size_bytes, transaction_id, rx_channel_id);
        }
    } else if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::LowLatencyRoutingFields>) {
        const uint32_t routing = cached_routing_fields.value & LowLatencyFields::FIELD_MASK;
        uint16_t payload_size_bytes = packet_start->payload_size_bytes;
        switch (routing) {
            case LowLatencyFields::WRITE_ONLY:
                generic_rx_local_delivery(packet_start, payload_size_bytes, transaction_id, rx_channel_id);
                break;
            case LowLatencyFields::FORWARD_ONLY: {
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                    packet_start, payload_size_bytes, cached_routing_fields, downstream_edm_interface, transaction_id);
                // In 1D, the forwarding sender channel is always sender channel 1
                // (channel 0 is worker, channel 1 receives forwarded traffic from upstream)
                channel_trimming_usage_recorder.set_sender_channel_forwarded_to(rx_channel_id, 1);
            } break;
            case LowLatencyFields::WRITE_AND_FORWARD: {
                // Resolve noc_send_type via the same packed 4B load the local-write path uses, then
                // reuse it through the _impl entry point so the standard path does no extra L1 read
                // versus the non-sparse build. The sparse case is unlikely, keeping the common path hot.
                const auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_start);
                if (packed.noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE) [[unlikely]] {
                    // This chip writes its own page group (counts[chip_idx] pages from write_idx), then
                    // write_idx/chip_idx advance so the downstream chip picks the next group. The advance
                    // must precede the forward because forwarding sends the (mutated) header downstream,
                    // and must follow the write because the write reads the pre-advance chip_idx.
                    execute_chip_unicast_to_local_chip_impl(
                        packet_start, payload_size_bytes, packed.noc_send_type, transaction_id, rx_channel_id);
                    auto& sparse = packet_start->command_fields.sparse_mcast_write;
                    sparse.write_idx += sparse.counts[sparse.chip_idx];
                    sparse.chip_idx++;
                    forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                        packet_start,
                        payload_size_bytes,
                        cached_routing_fields,
                        downstream_edm_interface,
                        transaction_id);
                } else {
                    forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                        packet_start,
                        payload_size_bytes,
                        cached_routing_fields,
                        downstream_edm_interface,
                        transaction_id);
                    execute_chip_unicast_to_local_chip_impl(
                        packet_start, payload_size_bytes, packed.noc_send_type, transaction_id, rx_channel_id);
                }
                channel_trimming_usage_recorder.set_sender_channel_forwarded_to(rx_channel_id, 1);
            } break;
            default: {
                ASSERT(false);
            }
        }
    }
}
#endif  // !FABRIC_2D

#if defined(FABRIC_2D)
template <typename D>
tt::tt_fabric::PacketHeaderBase<D>* packet_header_base_of(tt::tt_fabric::PacketHeaderBase<D>*);
using PacketHeaderBaseT =
    std::remove_pointer_t<decltype(packet_header_base_of(static_cast<PACKET_HEADER_TYPE*>(nullptr)))>;

struct PackedHeaderWord40 {
    uint16_t payload_size_bytes;
    tt::tt_fabric::NocSendType noc_send_type;
    uint8_t src_ch_id;

    static FORCE_INLINE PackedHeaderWord40 load(const volatile PACKET_HEADER_TYPE* hdr) {
        static_assert(offsetof(PacketHeaderBaseT, payload_size_bytes) == 40);
        static_assert(offsetof(PacketHeaderBaseT, noc_send_type) == 42);
        static_assert(offsetof(PacketHeaderBaseT, src_ch_id) == 43);
        const uint32_t raw = *reinterpret_cast<const volatile uint32_t*>(
            reinterpret_cast<uintptr_t>(hdr) + offsetof(PacketHeaderBaseT, payload_size_bytes));
        return {
            static_cast<uint16_t>(raw & 0xFFFFu),
            static_cast<tt::tt_fabric::NocSendType>((raw >> 16) & 0xFFu),
            static_cast<uint8_t>(raw >> 24)};
    }
};

// Helper to forward packet to local destination
// (relay in UDM mode, or local chip directly in non-UDM mode)
template <uint8_t rx_channel_id, typename LocalRelayInterfaceT>
FORCE_INLINE void forward_to_local_destination(
    LocalRelayInterfaceT& local_relay_interface,
    tt_l1_ptr PACKET_HEADER_TYPE* packet_start,
    uint16_t payload_size_bytes,
    tt::tt_fabric::NocSendType noc_send_type,
    uint8_t transaction_id,
    uint32_t* pending_atomic_flush = nullptr) {
    if constexpr (udm_mode) {
        execute_chip_unicast_to_relay(
            local_relay_interface, packet_start, payload_size_bytes, transaction_id, rx_channel_id);
    } else {
        execute_chip_unicast_to_local_chip_impl(
            packet_start, payload_size_bytes, noc_send_type, transaction_id, rx_channel_id, pending_atomic_flush);
    }
}

template <size_t DOWNSTREAM_EDM_SIZE>
constexpr uint8_t live_eth_mask_2d() {
    constexpr auto my_dir = static_cast<eth_chan_directions>(my_direction);
    constexpr auto dirs = Routing2DCodec::fwd_dirs<my_dir>();
    uint8_t live = 0;
    if constexpr (dispatch_arm_is_realizable<DOWNSTREAM_EDM_SIZE, dirs[0]>()) {
        live |= static_cast<uint8_t>(1u << 0);
    }
    if constexpr (dispatch_arm_is_realizable<DOWNSTREAM_EDM_SIZE, dirs[1]>()) {
        live |= static_cast<uint8_t>(1u << 1);
    }
    if constexpr (dispatch_arm_is_realizable<DOWNSTREAM_EDM_SIZE, dirs[2]>()) {
        live |= static_cast<uint8_t>(1u << 2);
    }
    if constexpr (dispatch_arm_is_realizable<DOWNSTREAM_EDM_SIZE, dirs[3]>()) {
        live |= static_cast<uint8_t>(1u << 3);
    }
    constexpr bool trim_ns_for_intramesh_ew =
        (my_dir == eth_chan_directions::EAST || my_dir == eth_chan_directions::WEST) && !is_intermesh_router;
    if constexpr (trim_ns_for_intramesh_ew) {
        live = static_cast<uint8_t>(live & ~0b0110);
    }
    return live;
}

// LIVE slot i is compact index i: fwd_dirs is enum order with self removed, matching
// direction_to_compact_index_map and host direction_compact_index. The credit-read helper
// is compact-keyed, so this identity is what keeps admit's stream id on the same slot init wrote.
constexpr bool live_slot_is_compact_index() {
    constexpr auto my_dir = static_cast<eth_chan_directions>(my_direction);
    constexpr auto dirs = Routing2DCodec::fwd_dirs<my_dir>();
    return get_downstream_edm_interface_index<dirs[0]>() == 0 && get_downstream_edm_interface_index<dirs[1]>() == 1 &&
           get_downstream_edm_interface_index<dirs[2]>() == 2 && get_downstream_edm_interface_index<dirs[3]>() == 3;
}
static_assert(live_slot_is_compact_index());

template <size_t DOWNSTREAM_EDM_SIZE>
constexpr bool use_2d_single_live_stateful_noc() {
    constexpr auto my_dir = static_cast<eth_chan_directions>(my_direction);
    constexpr bool is_ew = my_dir == eth_chan_directions::EAST || my_dir == eth_chan_directions::WEST;
    constexpr uint8_t LIVE = live_eth_mask_2d<DOWNSTREAM_EDM_SIZE>();
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED) && !defined(FABRIC_2D_VC1_ACTIVE)
    return is_ew && LIVE == 0b0001 && !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !udm_mode && !is_intermesh_router &&
           !is_intermesh_router_on_edge && !is_intramesh_router_on_edge;
#else
    return false;
#endif
}

// Checks local relay capacity when ld, plus the downstream queue for every LIVE direction whose
// action bit is set. Only UDM mode queues local delivery through a relay interface.
template <typename DownstreamSenderT, typename LocalRelayInterfaceT, size_t DOWNSTREAM_EDM_SIZE>
FORCE_INLINE bool admit_2d_dispatch(
    uint8_t action,
    bool ld,
    [[maybe_unused]] std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    [[maybe_unused]] LocalRelayInterfaceT& local_relay_interface) {
    constexpr uint8_t LIVE = live_eth_mask_2d<DOWNSTREAM_EDM_SIZE>();
    constexpr auto dirs = Routing2DCodec::fwd_dirs<static_cast<eth_chan_directions>(my_direction)>();
#if !defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)
    if constexpr (!udm_mode) {
        if constexpr (LIVE == 0b0001) {
            if (action & Routing2DCodec::action_bit(dirs[0])) {
                constexpr auto edm_index = get_downstream_edm_interface_index<dirs[0]>();
                return get_ptr_val<vc0_local_free_slots_stream_id<edm_index>()>() != 0;
            }
            return true;
        } else {
            if ((action & Routing2DCodec::ACTION_ETH_MASK) == 0) {
                return true;
            }
            [[maybe_unused]] int32_t free_0 = 0;
            [[maybe_unused]] int32_t free_1 = 0;
            [[maybe_unused]] int32_t free_2 = 0;
            [[maybe_unused]] int32_t free_3 = 0;
            if constexpr ((LIVE >> 0) & 1) {
                free_0 = get_ptr_val<vc0_local_free_slots_stream_id<get_downstream_edm_interface_index<dirs[0]>()>()>();
            }
            if constexpr ((LIVE >> 1) & 1) {
                free_1 = get_ptr_val<vc0_local_free_slots_stream_id<get_downstream_edm_interface_index<dirs[1]>()>()>();
            }
            if constexpr ((LIVE >> 2) & 1) {
                free_2 = get_ptr_val<vc0_local_free_slots_stream_id<get_downstream_edm_interface_index<dirs[2]>()>()>();
            }
            if constexpr ((LIVE >> 3) & 1) {
                free_3 = get_ptr_val<vc0_local_free_slots_stream_id<get_downstream_edm_interface_index<dirs[3]>()>()>();
            }
            bool ok = true;
            if constexpr ((LIVE >> 0) & 1) {
                if (action & Routing2DCodec::action_bit(dirs[0])) {
                    ok = ok && (free_0 != 0);
                }
            }
            if constexpr ((LIVE >> 1) & 1) {
                if (action & Routing2DCodec::action_bit(dirs[1])) {
                    ok = ok && (free_1 != 0);
                }
            }
            if constexpr ((LIVE >> 2) & 1) {
                if (action & Routing2DCodec::action_bit(dirs[2])) {
                    ok = ok && (free_2 != 0);
                }
            }
            if constexpr ((LIVE >> 3) & 1) {
                if (action & Routing2DCodec::action_bit(dirs[3])) {
                    ok = ok && (free_3 != 0);
                }
            }
            return ok;
        }
    }
#endif
    bool ok = true;
    if constexpr (udm_mode) {
        ok = !ld || local_relay_interface.template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>();
    }
    if constexpr ((LIVE >> 0) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[0])) {
            ok = ok && downstreams_have_space<DownstreamSenderT, LocalRelayInterfaceT, DOWNSTREAM_EDM_SIZE, dirs[0]>(
                           downstream_edm_interfaces, local_relay_interface);
        }
    }
    if constexpr ((LIVE >> 1) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[1])) {
            ok = ok && downstreams_have_space<DownstreamSenderT, LocalRelayInterfaceT, DOWNSTREAM_EDM_SIZE, dirs[1]>(
                           downstream_edm_interfaces, local_relay_interface);
        }
    }
    if constexpr ((LIVE >> 2) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[2])) {
            ok = ok && downstreams_have_space<DownstreamSenderT, LocalRelayInterfaceT, DOWNSTREAM_EDM_SIZE, dirs[2]>(
                           downstream_edm_interfaces, local_relay_interface);
        }
    }
    if constexpr ((LIVE >> 3) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[3])) {
            ok = ok && downstreams_have_space<DownstreamSenderT, LocalRelayInterfaceT, DOWNSTREAM_EDM_SIZE, dirs[3]>(
                           downstream_edm_interfaces, local_relay_interface);
        }
    }
    return ok;
}

// One unmodified full-packet copy per LIVE direction whose action bit is set. Remote receiver
// credit is left to the sender step's bubble flow control, and local delivery to the caller.
template <typename DownstreamSenderT, size_t DOWNSTREAM_EDM_SIZE>
FORCE_INLINE uint8_t forward_2d_dispatch(
    uint8_t action,
    tt_l1_ptr PACKET_HEADER_TYPE* packet_start,
    ROUTING_FIELDS_TYPE cached_routing_fields,
    std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    uint8_t transaction_id) {
    constexpr uint8_t LIVE = live_eth_mask_2d<DOWNSTREAM_EDM_SIZE>();
    constexpr auto dirs = Routing2DCodec::fwd_dirs<static_cast<eth_chan_directions>(my_direction)>();
    constexpr bool stateful_api = use_2d_single_live_stateful_noc<DOWNSTREAM_EDM_SIZE>();
    const uint16_t payload_size_bytes = packet_start->payload_size_bytes;
    uint8_t forwarded_slot_mask = 0;
    if constexpr ((LIVE >> 0) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[0])) {
            constexpr auto edm_index = get_downstream_edm_interface_index<dirs[0]>();
            forward_payload_to_downstream_edm<enable_deadlock_avoidance, stateful_api>(
                packet_start,
                payload_size_bytes,
                cached_routing_fields,
                downstream_edm_interfaces[edm_index],
                transaction_id);
            forwarded_slot_mask |= 1u << 0;
        }
    }
    if constexpr ((LIVE >> 1) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[1])) {
            constexpr auto edm_index = get_downstream_edm_interface_index<dirs[1]>();
            forward_payload_to_downstream_edm<enable_deadlock_avoidance, stateful_api>(
                packet_start,
                payload_size_bytes,
                cached_routing_fields,
                downstream_edm_interfaces[edm_index],
                transaction_id);
            forwarded_slot_mask |= 1u << 1;
        }
    }
    if constexpr ((LIVE >> 2) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[2])) {
            constexpr auto edm_index = get_downstream_edm_interface_index<dirs[2]>();
            forward_payload_to_downstream_edm<enable_deadlock_avoidance, stateful_api>(
                packet_start,
                payload_size_bytes,
                cached_routing_fields,
                downstream_edm_interfaces[edm_index],
                transaction_id);
            forwarded_slot_mask |= 1u << 2;
        }
    }
    if constexpr ((LIVE >> 3) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[3])) {
            constexpr auto edm_index = get_downstream_edm_interface_index<dirs[3]>();
            forward_payload_to_downstream_edm<enable_deadlock_avoidance, stateful_api>(
                packet_start,
                payload_size_bytes,
                cached_routing_fields,
                downstream_edm_interfaces[edm_index],
                transaction_id);
            forwarded_slot_mask |= 1u << 3;
        }
    }
    return forwarded_slot_mask;
}

// The host-provided slot map translates this receiver's compact downstream indices into each
// destination router's absolute sender channels. The row remains the receiving VC, including
// the VC0-to-VC1 crossover case.
template <uint8_t rx_channel_id>
FORCE_INLINE void record_2d_forwarded_slots(uint8_t forwarded_slot_mask) {
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
#if defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)
        constexpr uint32_t packed_downstream_sender_channel_ids = PACKED_DOWNSTREAM_VC1_SENDER_CHANNEL_IDS;
#else
        constexpr uint32_t packed_downstream_sender_channel_ids =
            rx_channel_id == 0 ? PACKED_DOWNSTREAM_VC0_SENDER_CHANNEL_IDS : PACKED_DOWNSTREAM_VC1_SENDER_CHANNEL_IDS;
#endif
        const uint16_t sender_mask =
            tt::tt_fabric::forwarded_slots_to_sender_mask(forwarded_slot_mask, packed_downstream_sender_channel_ids);
        channel_trimming_usage_recorder.merge_sender_channel_forwarded_to(rx_channel_id, sender_mask);
    }
}
#endif

template <typename EdmChannelWorkerIFs>
FORCE_INLINE void establish_edm_connection(
    EdmChannelWorkerIFs& local_sender_channel_worker_interface, [[maybe_unused]] uint32_t stream_id) {
    local_sender_channel_worker_interface
        .template cache_producer_noc_addr<ENABLE_RISC_CPU_DATA_CACHE, USE_DYNAMIC_CREDIT_ADDR>();
}

bool any_sender_channels_active(
    const std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids) {
    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        // The table is fabric-flat: this router's compact channel i sits at its fabric position.
        if (get_ptr_val(local_sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(i)]) !=
            static_cast<int32_t>(SENDER_NUM_BUFFERS_ARRAY[i])) {
            return true;
        }
    }
    return false;
}

constexpr bool IS_RETRAIN_SYNC_MASTER() { return enable_context_switch; }

// Signals intent to enter coordinated context switch and waits for erisc1 acknowledgement.
// After this returns, erisc1 is in a wait loop and will not execute routing logic.
FORCE_INLINE void coordinated_context_switch_start_as_master(
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    if constexpr (IS_RETRAIN_SYNC_MASTER()) {
        write_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>(
            static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::RETRAIN_INTENT));
        while (read_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>() !=
               static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::INTENT_ACK)) {
            if (got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
                return;
            }
        }
    }
}

// Signals completion of coordinated context switch and waits for erisc1 acknowledgement.
// After this returns, erisc1 is free to resume normal routing execution.
FORCE_INLINE void coordinated_context_switch_finish_as_master(
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    if constexpr (IS_RETRAIN_SYNC_MASTER()) {
        write_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>(
            static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::RETRAIN_COMPLETE));
        while (read_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>() !=
               static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::COMPLETE_ACK)) {
            if (got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
                return;
            }
        }
        write_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>(
            static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::NORMAL_EXECUTION));
    }
}

void run_routing_without_noc_sync_coordinated_as_master(
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    if constexpr (IS_RETRAIN_SYNC_MASTER()) {
        coordinated_context_switch_start_as_master(termination_signal_ptr);
        run_routing_without_noc_sync();
        coordinated_context_switch_finish_as_master(termination_signal_ptr);
    }
}

FORCE_INLINE void run_routing_without_noc_sync_coordinated_as_non_master(
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    if constexpr (!IS_RETRAIN_SYNC_MASTER()) {
        if (read_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>() ==
            static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::RETRAIN_INTENT)) {
            write_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>(
                static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::INTENT_ACK));
            while (read_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>() !=
                   static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::RETRAIN_COMPLETE)) {
                if (got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
                    return;
                }
            }
            write_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>(
                static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::COMPLETE_ACK));
            while (read_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>() !=
                   static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::NORMAL_EXECUTION)) {
                if (got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
                    return;
                }
            }
        }
    }
}

void run_coordinated_context_switch_to_base_firmware(
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    if constexpr (NUM_ACTIVE_ERISCS == 1) {
        run_routing_without_noc_sync();
    } else {
        if constexpr (IS_RETRAIN_SYNC_MASTER()) {
            run_routing_without_noc_sync_coordinated_as_master(termination_signal_ptr);
        }
    }
}

template <typename LocalTelemetryT>
FORCE_INLINE void update_telemetry(
    const std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids_ordered,
    bool tx_progress,
    bool rx_progress,
    LocalTelemetryT& local_fabric_telemetry,
    volatile tt_l1_ptr LocalTelemetryT* fabric_telemetry) {
    if constexpr (FABRIC_TELEMETRY_ROUTER_STATE) {
        const auto* routing_table_l1 =
            reinterpret_cast<tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
        auto* state_manager_l1 = const_cast<tt_l1_ptr RouterStateManager*>(&routing_table_l1->state_manager);
        fabric_telemetry->dynamic_info.erisc[MY_ERISC_ID].router_state = state_manager_l1->state;
    }
    if constexpr (FABRIC_TELEMETRY_HEARTBEAT_TX) {
        bool sender_idle = false;
        if (!tx_progress) {
            sender_idle = !any_sender_channels_active(local_sender_channel_free_slots_stream_ids_ordered);
        }
        if (tx_progress || sender_idle) {
            volatile RiscTimestampV2* tx_heartbeat_addr =
                &fabric_telemetry->dynamic_info.erisc[MY_ERISC_ID].tx_heartbeat;
            local_fabric_telemetry.dynamic_info.erisc[MY_ERISC_ID].tx_heartbeat.full++;
            tx_heartbeat_addr->full = local_fabric_telemetry.dynamic_info.erisc[MY_ERISC_ID].tx_heartbeat.full;
        }
    }
    if constexpr (FABRIC_TELEMETRY_HEARTBEAT_RX) {
        bool receiver_idle = false;
        if (!rx_progress) {
            receiver_idle = (get_ptr_val<to_receiver_packets_sent_streams[0]>() == 0);
        }
        if (rx_progress || receiver_idle) {
            volatile RiscTimestampV2* rx_heartbeat_addr =
                &fabric_telemetry->dynamic_info.erisc[MY_ERISC_ID].rx_heartbeat;
            local_fabric_telemetry.dynamic_info.erisc[MY_ERISC_ID].rx_heartbeat.full++;
            rx_heartbeat_addr->full = local_fabric_telemetry.dynamic_info.erisc[MY_ERISC_ID].rx_heartbeat.full;
        }
    }

    if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
        // Helper to safely write to volatile BandwidthTelemetry destinations without discarding qualifiers
        auto store_bandwidth_telemetry = [](volatile BandwidthTelemetry* dst, const BandwidthTelemetry& src) {
            dst->elapsed_active_cycles.full = src.elapsed_active_cycles.full;
            dst->elapsed_cycles.full = src.elapsed_cycles.full;
            dst->num_words_sent = src.num_words_sent;
            dst->num_packets_sent = src.num_packets_sent;
        };

        if constexpr (NUM_ACTIVE_ERISCS == 1) {
            store_bandwidth_telemetry(
                &fabric_telemetry->dynamic_info.tx_bandwidth, local_fabric_telemetry.dynamic_info.tx_bandwidth);
            store_bandwidth_telemetry(
                &fabric_telemetry->dynamic_info.rx_bandwidth, local_fabric_telemetry.dynamic_info.rx_bandwidth);
        } else {
            if constexpr (MY_ERISC_ID == 0) {
                store_bandwidth_telemetry(
                    &fabric_telemetry->dynamic_info.tx_bandwidth, local_fabric_telemetry.dynamic_info.tx_bandwidth);
            } else {
                store_bandwidth_telemetry(
                    &fabric_telemetry->dynamic_info.rx_bandwidth, local_fabric_telemetry.dynamic_info.rx_bandwidth);
            }
        }
    }
}

template <bool enable_deadlock_avoidance, bool SKIP_CONNECTION_LIVENESS_CHECK, typename EdmChannelWorkerIFs>
FORCE_INLINE void send_credits_to_upstream_workers(
    EdmChannelWorkerIFs& local_sender_channel_worker_interface,
    int32_t num_credits,
    bool channel_connection_established) {
    if constexpr (SKIP_CONNECTION_LIVENESS_CHECK) {
        local_sender_channel_worker_interface
            .template notify_persistent_connection_of_free_space<enable_deadlock_avoidance>(num_credits);
    } else {
        // Connection liveness checks are only done for connections that are not persistent
        // For those connections, it's unsafe to use free-slots counters held in stream registers
        // due to the lack of race avoidant connection protocol. Therefore, we update our read counter
        // instead because these connections will be read/write counter based instead
        local_sender_channel_worker_interface.increment_local_read_counter(num_credits);
        if (channel_connection_established) {
            local_sender_channel_worker_interface
                .template notify_worker_of_read_counter_update<enable_read_counter_update_noc_flush>();
        } else {
            local_sender_channel_worker_interface.copy_read_counter_to_worker_location_info();
        }
    }
}

template <typename LocalTelemetryT>
FORCE_INLINE void update_bw_counters(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header, LocalTelemetryT& local_fabric_telemetry) {
    if constexpr ((NUM_ACTIVE_ERISCS == 1) || (MY_ERISC_ID == 0)) {
        size_t packet_bytes = packet_header->get_payload_size_including_header();
        local_fabric_telemetry.dynamic_info.tx_bandwidth.num_packets_sent++;
        local_fabric_telemetry.dynamic_info.tx_bandwidth.num_words_sent += (packet_bytes + 3) >> 2;
    }
    if constexpr ((NUM_ACTIVE_ERISCS == 1) || (MY_ERISC_ID == 1)) {
        size_t packet_bytes = packet_header->get_payload_size_including_header();
        local_fabric_telemetry.dynamic_info.rx_bandwidth.num_packets_sent++;
        local_fabric_telemetry.dynamic_info.rx_bandwidth.num_words_sent += (packet_bytes + 3) >> 2;
    }
}

template <typename LocalTelemetryT>
FORCE_INLINE void update_bw_cycles(
    uint64_t loop_delta_cycles, bool tx_progress, bool rx_progress, LocalTelemetryT& local_fabric_telemetry) {
    if constexpr ((NUM_ACTIVE_ERISCS == 1) || (MY_ERISC_ID == 0)) {
        local_fabric_telemetry.dynamic_info.tx_bandwidth.elapsed_cycles.full += loop_delta_cycles;
        if (tx_progress) {
            local_fabric_telemetry.dynamic_info.tx_bandwidth.elapsed_active_cycles.full += loop_delta_cycles;
        }
    }
    if constexpr ((NUM_ACTIVE_ERISCS == 1) || (MY_ERISC_ID == 1)) {
        local_fabric_telemetry.dynamic_info.rx_bandwidth.elapsed_cycles.full += loop_delta_cycles;
        if (rx_progress) {
            local_fabric_telemetry.dynamic_info.rx_bandwidth.elapsed_active_cycles.full += loop_delta_cycles;
        }
    }
}

// ============================================================================
// Sender channels: grouped free-slot reads and idle-channel tests
// ============================================================================
// Each pass reads the free-slot stream registers of up to four VC0 sender channels as one group (the generic loop and
// sender_erisc alike), and each channel visit uses its value.
//
// Grouped reads. Register and L1 reads take many cycles but do not stall the core until their value is used, so
// reads issued back to back overlap. This file expresses a group as volatile loads between two
// std::atomic_signal_fence(seq_cst): no memory access moves into or out of the group and the loads stay in program
// order. Their values are used only after the closing fence. The fences do not stop the compiler from scheduling
// unrelated arithmetic between the loads.
constexpr size_t grouped_packet_check_channels = ACTUAL_VC0_SENDER_CHANNELS < 4 ? ACTUAL_VC0_SENDER_CHANNELS : 4;

// Channel CH's free-slots register, unmasked (the caller masks; get_ptr_val would mask at once).
template <size_t CH>
FORCE_INLINE uint32_t read_free_slots_raw() {
    return NOC_STREAM_READ_REG(
        sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(CH)],
        STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
}

template <size_t N = grouped_packet_check_channels>
FORCE_INLINE void read_grouped_free_slots(uint32_t (&raw)[4]) {
    if constexpr (N >= 1) {
        raw[0] = read_free_slots_raw<0>();
    }
    if constexpr (N >= 2) {
        raw[1] = read_free_slots_raw<1>();
    }
    if constexpr (N >= 3) {
        raw[2] = read_free_slots_raw<2>();
    }
    if constexpr (N >= 4) {
        raw[3] = read_free_slots_raw<3>();
    }
    // Closes the group (see "Grouped reads"). There is no opening fence: the group starts the pass.
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

// Idle forwarding channels (1..3) return at the top of their visit when the grouped value says every slot is free and
// every packet sent has completed. Applies when all serviced forwarding channels are in the group, skip the liveness
// check, use counter credits, and the VC has no first-level ack.
constexpr bool skip_idle_channel(size_t ch) {
    return ch >= 1 && ch < grouped_packet_check_channels && is_sender_channel_serviced[ch];
}
constexpr bool skip_idle_per_channel = [] {
    for (size_t ch = 1; ch <= (is_2d_fabric ? 3 : 1) && ch < MAX_NUM_SENDER_CHANNELS; ch++) {
        if (is_sender_channel_serviced[ch] && !skip_idle_channel(ch)) {
            return false;
        }
    }
    bool any = false;
    for (size_t ch = 1; ch < grouped_packet_check_channels; ch++) {
        if (is_sender_channel_serviced[ch]) {
            if (!sender_ch_live_check_skip[ch] || !sender_channel_uses_counter_credits(ch)) {
                return false;
            }
            any = true;
        }
    }
    return any && !ENABLE_FIRST_LEVEL_ACK_VC0;
}();

// With equal power-of-two slot counts the all-free test is one bit test. It equals the masked compare only while the
// register holds a count in [0, slots]; a register driven out of that range by a credit error can read as all-free.
constexpr bool skip_idle_bit_test = [] {
    size_t n = 0;
    for (size_t ch = 1; ch < grouped_packet_check_channels; ch++) {
        if (skip_idle_channel(ch)) {
            const size_t slots = SENDER_NUM_BUFFERS_ARRAY[ch];
            if ((slots & (slots - 1)) != 0 || slots >= (1u << REMOTE_DEST_WORDS_FREE_WIDTH) || (n != 0 && slots != n)) {
                return false;
            }
            n = slots;
        }
    }
    return n != 0;
}();

// The count in a stream credit register (packets received, free slots). A count never exceeds its channel's slot count
// (each user static_asserts its slot count against this mask), so the low 11 bits hold it.
constexpr uint32_t stream_count_mask = 0x7FFu;
static_assert(
    stream_count_mask <= ((1u << REMOTE_DEST_WORDS_FREE_WIDTH) - 1),
    "stream_count_mask must fit within REMOTE_DEST_WORDS_FREE_WIDTH");

template <size_t CH>
FORCE_INLINE bool channel_all_slots_free(uint32_t raw) {
    static_assert(SENDER_NUM_BUFFERS_ARRAY[CH] <= stream_count_mask);
    if constexpr (skip_idle_bit_test) {
        return (raw & SENDER_NUM_BUFFERS_ARRAY[CH]) != 0;
    } else {
        return (raw & stream_count_mask) == SENDER_NUM_BUFFERS_ARRAY[CH];
    }
}

// ============================================================================
// Sender and receiver channel steps
// ============================================================================
constexpr bool is_blackhole_arch =
#if defined(ARCH_BLACKHOLE)
    true;
#else
    false;
#endif

// After the data command, the send issues the TX-queue busy check and the channel's L1 completion-counter read
// together and overlaps them with the slot bookkeeping. Needs counter credits (an L1 counter to read) and a send that
// does not spin on the queue.
template <uint8_t sender_channel_index>
constexpr bool sender_channel_overlaps_send_completion() {
    return is_blackhole_arch && sender_channel_uses_counter_credits(sender_channel_index) &&
           !ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA && !ENABLE_RISC_CPU_DATA_CACHE;
}

// Sends the channel's next packet when it has one and the far receiver ring has room (two slots for an injection
// channel under bubble flow control). Reads the payload size and the TX-queue busy status together and stamps
// src_ch_id; a busy queue yields (returns false, no send). Otherwise issues the data command, advances the far slot
// pointer and the channel's slot, and increments the far receiver's packets-received register on the same queue.
template <
    uint8_t sender_channel_index,
    uint8_t to_receiver_pkts_sent_id,
    bool enable_first_level_ack,
    typename SenderChannelT,
    typename ReceiverPointersT,
    typename LocalTelemetryT,
    typename SenderChannelFromReceiverCreditsT>
FORCE_INLINE bool sender_channel_try_send(
    SenderChannelT& local_sender_channel,
    ReceiverPointersT& outbound_to_receiver_channel_pointers,
    SenderChannelFromReceiverCreditsT& sender_channel_from_receiver_credits,
    PerfTelemetryRecorder& perf_telemetry_recorder,
    LocalTelemetryT& local_fabric_telemetry,
    bool has_unsent_packet,
    int32_t& completions_since_last_check) {
    constexpr uint32_t sender_channel_free_slots_stream_id =
        sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(sender_channel_index)];
    constexpr bool use_bubble_flow_control =
        sender_channel_is_traffic_injection_channel[sender_channel_index] && enable_deadlock_avoidance;
    static_assert(
        !use_bubble_flow_control || enable_first_level_ack,
        "Bubble flow control and first level ack must be set to the same values");

    bool receiver_has_space_for_packet;
    if constexpr (use_bubble_flow_control) {
        receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.num_free_slots >=
                                        BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS;
    } else {
        receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet();
    }
    if (!receiver_has_space_for_packet || !has_unsent_packet) {
        return false;
    }

    const uint32_t src_addr = local_sender_channel.get_cached_next_buffer_slot_addr();
    auto* pkt_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(src_addr);
    constexpr uint32_t txq_base = ETH_TXQ0_REGS_START + sender_txq_id * ETH_TXQ_REGS_SIZE;
    volatile uint32_t* const txq_regs = reinterpret_cast<volatile uint32_t*>(txq_base);

    size_t payload_size_bytes;
    if constexpr (!ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA && is_blackhole_arch) {
        // Read the payload size and the TX-queue busy status (BH-55 dummy CMD read, then STATUS) as one group.
        std::atomic_signal_fence(std::memory_order_seq_cst);
        const uint32_t raw_payload_size = pkt_header->get_payload_size_excluding_header();
        (void)txq_regs[ETH_TXQ_CMD >> 2];
        const uint32_t pre_txq_status = txq_regs[ETH_TXQ_STATUS >> 2];
        std::atomic_signal_fence(std::memory_order_seq_cst);

        did_something = true;
        if constexpr (!skip_src_ch_id_update) {
            pkt_header->src_ch_id = sender_channel_index;
        }
        std::atomic_signal_fence(std::memory_order_seq_cst);

        if ((pre_txq_status >> ETH_TXQ_STATUS_CMD_ONGOING_BIT) & 1u) {
            return false;
        }
        payload_size_bytes = raw_payload_size + sizeof(PACKET_HEADER_TYPE);
    } else {
        payload_size_bytes = pkt_header->get_payload_size_including_header();
        if constexpr (!ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
            if (internal_::eth_txq_is_busy(sender_txq_id)) {
                return false;
            }
        }
        did_something = true;
        if constexpr (!skip_src_ch_id_update) {
            pkt_header->src_ch_id = sender_channel_index;
        }
        if constexpr (ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
            while (internal_::eth_txq_is_busy(sender_txq_id)) {
            }
        }
    }

    record_sender_channel_usage(sender_channel_index, payload_size_bytes);

    const auto dest_addr = outbound_to_receiver_channel_pointers.remote_receiver_channel_address_ptr;
    internal_::eth_send_packet_bytes_unsafe(sender_txq_id, src_addr, dest_addr, payload_size_bytes);

    // Advance receiver buffer pointers
    outbound_to_receiver_channel_pointers.advance_remote_receiver_buffer_pointer();
    local_sender_channel.advance_to_next_cached_buffer_slot_addr_by_stride();
    record_packet_send(perf_telemetry_recorder, sender_channel_index, payload_size_bytes);

    if constexpr (sender_channel_overlaps_send_completion<sender_channel_index>()) {
        constexpr uint32_t rem_stream_addr =
            STREAM_REG_ADDR(to_receiver_pkts_sent_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
        volatile tt_l1_ptr uint32_t* const cptr = sender_channel_from_receiver_credits.completions_received_counter_ptr;

        // Issue the TX-queue busy reads (BH-55 dummy CMD + STATUS) and the L1 completion-counter read back to back,
        // fenced so the bookkeeping below (free-slot count, stream-register increment) runs while they are in flight.
        std::atomic_signal_fence(std::memory_order_seq_cst);
        (void)txq_regs[ETH_TXQ_CMD >> 2];
        const uint32_t txq_status_raw = txq_regs[ETH_TXQ_STATUS >> 2];
        const uint32_t comp_raw = *cptr;
        std::atomic_signal_fence(std::memory_order_seq_cst);

        outbound_to_receiver_channel_pointers.num_free_slots--;
        increment_local_update_ptr_val<sender_channel_free_slots_stream_id>(1);
        std::atomic_signal_fence(std::memory_order_seq_cst);

        if ((txq_status_raw >> ETH_TXQ_STATUS_CMD_ONGOING_BIT) & 1u) {
            while (internal_::eth_txq_is_busy(sender_txq_id)) {
            }
        }
        txq_regs[ETH_TXQ_DEST_ADDR >> 2] = rem_stream_addr;
        txq_regs[ETH_TXQ_REMOTE_REG_DATA >> 2] = 1u << REMOTE_DEST_BUF_WORDS_FREE_INC;
        txq_regs[ETH_TXQ_CMD >> 2] = ETH_TXQ_CMD_START_REG;
        sender_channel_from_receiver_credits.note_packet_sent();
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            update_bw_counters(pkt_header, local_fabric_telemetry);
        }
        completions_since_last_check =
            static_cast<int32_t>(comp_raw - sender_channel_from_receiver_credits.completions_received_and_processed);
    } else {
        outbound_to_receiver_channel_pointers.num_free_slots--;
        while (internal_::eth_txq_is_busy(sender_txq_id)) {
        }
        remote_update_ptr_val<to_receiver_pkts_sent_id, sender_txq_id>(1U);

        sender_channel_from_receiver_credits.note_packet_sent();
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            update_bw_counters(pkt_header, local_fabric_telemetry);
        }
        increment_local_update_ptr_val<sender_channel_free_slots_stream_id>(1);
    }

    return true;
}

template <
    uint8_t sender_channel_index,
    bool SKIP_CONNECTION_LIVENESS_CHECK,
    bool enable_first_level_ack,
    typename WorkerInterfaceT,
    typename ReceiverPointersT,
    typename SenderChannelFromReceiverCreditsT>
FORCE_INLINE void sender_channel_process_completions(
    WorkerInterfaceT& local_sender_channel_worker_interface,
    ReceiverPointersT& outbound_to_receiver_channel_pointers,
    SenderChannelFromReceiverCreditsT& sender_channel_from_receiver_credits,
    bool channel_connection_established,
    bool sent_packet,
    int32_t completions_since_last_check) {
    if (!sender_channel_overlaps_send_completion<sender_channel_index>() || !sent_packet) {
        if (sender_channel_from_receiver_credits.has_completions_pending()) {
            completions_since_last_check =
                sender_channel_from_receiver_credits
                    .template get_num_unprocessed_completions_from_receiver<ENABLE_RISC_CPU_DATA_CACHE>();
        }
    }
    if (completions_since_last_check) {
        outbound_to_receiver_channel_pointers.num_free_slots += completions_since_last_check;
        sender_channel_from_receiver_credits.increment_num_processed_completions(completions_since_last_check);
        if constexpr (!enable_first_level_ack) {
            send_credits_to_upstream_workers<enable_deadlock_avoidance, SKIP_CONNECTION_LIVENESS_CHECK>(
                local_sender_channel_worker_interface, completions_since_last_check, channel_connection_established);
        }
    }

    if constexpr (enable_first_level_ack) {
        uint32_t acks_since_last_check = 0;
        if (sender_channel_from_receiver_credits.has_acks_pending()) {
            acks_since_last_check = sender_channel_from_receiver_credits
                                        .template get_num_unprocessed_acks_from_receiver<ENABLE_RISC_CPU_DATA_CACHE>();
        }
        if (acks_since_last_check > 0) {
            sender_channel_from_receiver_credits.increment_num_processed_acks(acks_since_last_check);
            send_credits_to_upstream_workers<enable_deadlock_avoidance, SKIP_CONNECTION_LIVENESS_CHECK>(
                local_sender_channel_worker_interface, acks_since_last_check, channel_connection_established);
        }
    }
}

template <
    uint8_t sender_channel_index,
    uint8_t to_receiver_pkts_sent_id,
    bool enable_first_level_ack,
    typename SenderChannelT,
    typename WorkerInterfaceT,
    typename ReceiverPointersT,
    typename LocalTelemetryT,
    typename SenderChannelFromReceiverCreditsT>
#if !defined(FABRIC_2D_VC1_ACTIVE)
FORCE_INLINE
#endif
    bool
    run_worker_sender_step(
        SenderChannelT& local_sender_channel,
        WorkerInterfaceT& local_sender_channel_worker_interface,
        ReceiverPointersT& outbound_to_receiver_channel_pointers,
        bool& channel_connection_established,
        uint32_t sender_channel_free_slots_stream_id,
        SenderChannelFromReceiverCreditsT& sender_channel_from_receiver_credits,
        PerfTelemetryRecorder& perf_telemetry_recorder,
        LocalTelemetryT& local_fabric_telemetry,
        uint32_t grouped_free_slots_raw = 0) {
    static_assert(WorkerInterfaceT::num_buffers <= stream_count_mask);
    uint32_t free_slots;
    if constexpr (sender_channel_index < grouped_packet_check_channels) {
        free_slots = grouped_free_slots_raw & stream_count_mask;
    } else {
        constexpr uint32_t stream_id =
            sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(sender_channel_index)];
        free_slots = get_ptr_val<stream_id>();
    }
    const bool has_unsent_packet = free_slots != WorkerInterfaceT::num_buffers;

    int32_t completions_since_last_check = 0;
    const bool progress =
        sender_channel_try_send<sender_channel_index, to_receiver_pkts_sent_id, enable_first_level_ack>(
            local_sender_channel,
            outbound_to_receiver_channel_pointers,
            sender_channel_from_receiver_credits,
            perf_telemetry_recorder,
            local_fabric_telemetry,
            has_unsent_packet,
            completions_since_last_check);

    sender_channel_process_completions<
        sender_channel_index,
        /*SKIP_CONNECTION_LIVENESS_CHECK=*/false,
        enable_first_level_ack>(
        local_sender_channel_worker_interface,
        outbound_to_receiver_channel_pointers,
        sender_channel_from_receiver_credits,
        channel_connection_established,
        progress,
        completions_since_last_check);

    // Worker connection open/teardown, polled on every visit.
    if (!channel_connection_established || local_sender_channel_worker_interface.has_worker_teardown_request()) {
        check_worker_connections<MY_ETH_CHANNEL, ENABLE_RISC_CPU_DATA_CACHE>(
            local_sender_channel_worker_interface, channel_connection_established, sender_channel_free_slots_stream_id);
    }
    return progress;
}

template <
    uint8_t sender_channel_index,
    uint8_t to_receiver_pkts_sent_id,
    bool enable_first_level_ack,
    typename SenderChannelT,
    typename WorkerInterfaceT,
    typename ReceiverPointersT,
    typename LocalTelemetryT,
    typename SenderChannelFromReceiverCreditsT>
#if !defined(FABRIC_2D_VC1_ACTIVE)
FORCE_INLINE
#endif
    bool
    run_forwarding_sender_step(
        SenderChannelT& local_sender_channel,
        WorkerInterfaceT& local_sender_channel_worker_interface,
        ReceiverPointersT& outbound_to_receiver_channel_pointers,
        [[maybe_unused]] uint32_t sender_channel_free_slots_stream_id,
        SenderChannelFromReceiverCreditsT& sender_channel_from_receiver_credits,
        PerfTelemetryRecorder& perf_telemetry_recorder,
        LocalTelemetryT& local_fabric_telemetry,
        uint32_t grouped_free_slots_raw = 0) {
    bool has_unsent_packet;
    if constexpr (skip_idle_per_channel && skip_idle_channel(sender_channel_index)) {
        const bool all_slots_free = channel_all_slots_free<sender_channel_index>(grouped_free_slots_raw);
        if (all_slots_free && !sender_channel_from_receiver_credits.has_completions_pending()) {
            return false;
        }
        has_unsent_packet = !all_slots_free;
    } else {
        static_assert(WorkerInterfaceT::num_buffers <= stream_count_mask);
        uint32_t free_slots;
        if constexpr (sender_channel_index < grouped_packet_check_channels) {
            free_slots = grouped_free_slots_raw & stream_count_mask;
        } else {
            constexpr uint32_t stream_id =
                sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(sender_channel_index)];
            free_slots = get_ptr_val<stream_id>();
        }
        has_unsent_packet = free_slots != WorkerInterfaceT::num_buffers;
    }

    int32_t completions_since_last_check = 0;
    const bool progress =
        sender_channel_try_send<sender_channel_index, to_receiver_pkts_sent_id, enable_first_level_ack>(
            local_sender_channel,
            outbound_to_receiver_channel_pointers,
            sender_channel_from_receiver_credits,
            perf_telemetry_recorder,
            local_fabric_telemetry,
            has_unsent_packet,
            completions_since_last_check);

    sender_channel_process_completions<
        sender_channel_index,
        /*SKIP_CONNECTION_LIVENESS_CHECK=*/true,
        enable_first_level_ack>(
        local_sender_channel_worker_interface,
        outbound_to_receiver_channel_pointers,
        sender_channel_from_receiver_credits,
        /*channel_connection_established=*/true,
        progress,
        completions_since_last_check);

    return progress;
}

template <
    uint8_t VC_RECEIVER_CHANNEL,
    uint8_t sender_channel_index,
    bool enable_first_level_ack,
    typename EthSenderChannels,
    typename EdmChannelWorkerIFs,
    typename ReceiverPointersT,
    size_t NUM_SENDER_CHANNELS,
    typename LocalTelemetryT>
#if !defined(FABRIC_2D_VC1_ACTIVE)
FORCE_INLINE
#endif
    bool
    run_sender_channel_step(
        EthSenderChannels& local_sender_channels,
        EdmChannelWorkerIFs& local_sender_channel_worker_interfaces,
        ReceiverPointersT& outbound_to_receiver_channel_pointers,
        std::array<bool, NUM_SENDER_CHANNELS>& channel_connection_established,
        std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids,
        SenderChannelFromReceiverCredits<NUM_SENDER_CHANNELS>& sender_channel_from_receiver_credits,
        PerfTelemetryRecorder& perf_telemetry_recorder,
        LocalTelemetryT& local_fabric_telemetry,
        uint32_t grouped_free_slots_raw = 0) {
    if constexpr (is_sender_channel_serviced[sender_channel_index]) {
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();

        bool progress;
        if constexpr (sender_ch_live_check_skip[sender_channel_index]) {
            progress = run_forwarding_sender_step<
                sender_channel_index,
                to_receiver_packets_sent_streams[VC_RECEIVER_CHANNEL],
                enable_first_level_ack>(
                local_sender_channels.template get<sender_channel_index>(),
                local_sender_channel_worker_interfaces.template get<sender_channel_index>(),
                outbound_to_receiver_channel_pointers,
                local_sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(sender_channel_index)],
                sender_channel_from_receiver_credits.template get<sender_channel_index>(),
                perf_telemetry_recorder,
                local_fabric_telemetry,
                grouped_free_slots_raw);
        } else {
            progress = run_worker_sender_step<
                sender_channel_index,
                to_receiver_packets_sent_streams[VC_RECEIVER_CHANNEL],
                enable_first_level_ack>(
                local_sender_channels.template get<sender_channel_index>(),
                local_sender_channel_worker_interfaces.template get<sender_channel_index>(),
                outbound_to_receiver_channel_pointers,
                channel_connection_established[sender_channel_index],
                local_sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(sender_channel_index)],
                sender_channel_from_receiver_credits.template get<sender_channel_index>(),
                perf_telemetry_recorder,
                local_fabric_telemetry,
                grouped_free_slots_raw);
        }
        return progress;
    }
    return false;
}

template <
    uint8_t receiver_channel,
    uint8_t to_receiver_pkts_sent_id,
    bool enable_first_level_ack,
    size_t DOWNSTREAM_EDM_SIZE,
    typename WriteTridTracker,
    typename ReceiverChannelBufferT,
    typename ReceiverChannelPointersT,
    typename DownstreamSenderT,
    typename LocalRelayInterfaceT,
    typename LocalTelemetryT,
    typename ReceiverChannelResponseCreditSenderT>
FORCE_INLINE bool run_receiver_channel_step_impl(
    ReceiverChannelBufferT& local_receiver_channel,
    [[maybe_unused]] std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    [[maybe_unused]] LocalRelayInterfaceT& local_relay_interface,
    ReceiverChannelPointersT& receiver_channel_pointers,
    WriteTridTracker& receiver_channel_trid_tracker,
    [[maybe_unused]] std::array<uint8_t, num_eth_ports>& port_direction_table,
    ReceiverChannelResponseCreditSenderT& receiver_channel_response_credit_sender,
    [[maybe_unused]] const tt::tt_fabric::routing_l1_info_t& routing_table,
    [[maybe_unused]] uint16_t my_mesh_id,
    [[maybe_unused]] uint8_t my_mesh_coord_y,
    [[maybe_unused]] uint8_t my_mesh_coord_x,
    LocalTelemetryT& local_fabric_telemetry,
    [[maybe_unused]] uint32_t& rx_idle_steps,
    [[maybe_unused]] uint32_t& pending_atomic_flush) {
    bool progress = false;
    auto& wr_sent_counter = receiver_channel_pointers.wr_sent_counter;
    int32_t pkts_received_since_last_check = get_ptr_val<to_receiver_pkts_sent_id>();

    bool unwritten_packets;
    if constexpr (enable_first_level_ack) {
        auto& ack_counter = receiver_channel_pointers.ack_counter;
        bool pkts_received = pkts_received_since_last_check > 0;
        bool can_send_ack = pkts_received;
        if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
            can_send_ack = can_send_ack && !internal_::eth_txq_is_busy(receiver_txq_id);
        }
        if (can_send_ack) {
            router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
            increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);

            uint8_t src_ch_id;
            if constexpr (skip_src_ch_id_update) {
                src_ch_id = receiver_channel_pointers.get_src_chan_id();
            } else {
                auto receiver_buffer_index = ack_counter.get_buffer_index();
                tt_l1_ptr PACKET_HEADER_TYPE* packet_header = const_cast<PACKET_HEADER_TYPE*>(
                    local_receiver_channel.template get_packet_header<PACKET_HEADER_TYPE>(receiver_buffer_index));
                receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);
                src_ch_id = receiver_channel_pointers.get_src_chan_id(receiver_buffer_index);
            }

            receiver_send_received_ack<ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK>(
                receiver_channel_response_credit_sender, src_ch_id);
            ack_counter.increment();
        }
        unwritten_packets = !wr_sent_counter.is_caught_up_to(ack_counter);
    } else {
        unwritten_packets = pkts_received_since_last_check != 0;
    }

    NamedProfiler<
        CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        receiver_forward_timer;
    receiver_forward_timer.set_should_dump(unwritten_packets);
    receiver_forward_timer.open();

    if (unwritten_packets) {
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr PACKET_HEADER_TYPE* packet_header = const_cast<PACKET_HEADER_TYPE*>(
            local_receiver_channel.template get_packet_header<PACKET_HEADER_TYPE>(receiver_buffer_index));

        ROUTING_FIELDS_TYPE cached_routing_fields{};
#ifndef FABRIC_2D
        cached_routing_fields = packet_header->routing_fields;
        if constexpr (!skip_src_ch_id_update && !enable_first_level_ack) {
            receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);
        }
#else
        const auto hdr_w40 = PackedHeaderWord40::load(packet_header);
        if constexpr (!skip_src_ch_id_update && !enable_first_level_ack) {
            receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, hdr_w40.src_ch_id);
        }
#endif
        // Decode action is the dispatch word: LIVE slots test action_bit(dirs[slot]). Local
        // delivery stays a separate flag (bit 5).
        uint8_t action = 0;
        bool local_deliver = false;
        // Set only when this chip is the mesh's exit for the packet; carries the resolved INTERMESH
        // downstream slot from admission to the forward phase.
        bool intermesh_egress = false;
        uint8_t intermesh_egress_index = 0;
        bool can_send_to_all_local_chip_receivers;
        if constexpr (is_2d_fabric) {
#if defined(FABRIC_2D)
            // 2D action-map RX path: landing intercept (boundary receivers only) -> decode ->
            // intermesh-exit intercept -> linear LIVE admit on action bits. No hop program
            // and no header mutation. Self is not in fwd_dirs so no arm tests it; extra bits
            // outside LIVE are ignored. boundary_dir_is_addressable is the only remaining
            // bounds check, because a bad intermesh exit index walks off the compact array.
            if constexpr (receiver_channel_is_intermesh_ingress[receiver_channel]) {
                // Runs before decode, which would otherwise consume stale source-mesh action maps. The
                // landing encode replaces them from this mesh's destination-major route table.
                fabric_set_2d_intermesh_landing_route(packet_header, routing_table, MESH_Y_SIZE, MESH_X_SIZE);
                // Rebuilt map requires Y action decode first.
                action = Routing2DCodec::decode_action_y_first(
                    packet_header->route_buffer, my_mesh_coord_y, my_mesh_coord_x, MESH_Y_SIZE);
            } else {
                action = Routing2DCodec::decode_action<static_cast<eth_chan_directions>(my_direction)>(
                    packet_header->route_buffer, my_mesh_coord_y, my_mesh_coord_x, MESH_Y_SIZE);
            }
            // This chip is the exit when the maps say deliver here but the final mesh is
            // elsewhere. CT-gated so interior routers skip the mesh-id compare.
            bool intermesh_exit = false;
            if constexpr (is_intramesh_router_on_edge) {
                intermesh_exit =
                    Routing2DCodec::action_is_intermesh_exit(action, packet_header->dst_start_mesh_id, my_mesh_id);
            }
            if (intermesh_exit) {
                // Forward the packet as-is on the INTERMESH egress in boundary_dir, with no
                // local delivery and no map rebuild.
                const auto boundary_dir = static_cast<eth_chan_directions>(
                    routing_table.inter_mesh_direction_table.get_original_direction(
                        packet_header->dst_start_mesh_id));
                // boundary_dir is chosen per destination mesh, so unlike a decoded action it is
                // not constrained to the arms this router was built with. Indexing the compact
                // array unchecked would read past its end, so fail-stop instead.
                const bool boundary_dir_is_addressable =
                    boundary_dir < eth_chan_directions::COUNT &&
                    boundary_dir != static_cast<eth_chan_directions>(my_direction) &&
                    get_downstream_edm_interface_index(boundary_dir) < downstream_edm_interfaces.size();
                if (!boundary_dir_is_addressable) {
                    ASSERT(false);
                    can_send_to_all_local_chip_receivers = false;
                } else {
                    intermesh_egress = true;
                    intermesh_egress_index = static_cast<uint8_t>(get_downstream_edm_interface_index(boundary_dir));
                    can_send_to_all_local_chip_receivers =
                        downstream_edm_interfaces[intermesh_egress_index]
                            .template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>();
                }
            } else {
                local_deliver = (action & Routing2DCodec::ACTION_LOCAL_DELIVER) != 0;
                can_send_to_all_local_chip_receivers =
                    admit_2d_dispatch(action, local_deliver, downstream_edm_interfaces, local_relay_interface);
            }
#endif
        } else {
#ifndef FABRIC_2D
            can_send_to_all_local_chip_receivers =
                can_forward_packet_completely(cached_routing_fields, downstream_edm_interfaces[0]);
#endif
        }
        if constexpr (enable_trid_flush_check_on_noc_txn) {
            bool trid_flushed = receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
            can_send_to_all_local_chip_receivers &= trid_flushed;
        }
        if (can_send_to_all_local_chip_receivers) {
            did_something = true;
            progress = true;
            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                update_bw_counters(packet_header, local_fabric_telemetry);
            }
            channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(receiver_channel);
            if constexpr (!is_receiver_channel_forwarding_disabled[receiver_channel]) {
                uint8_t trid =
                    receiver_channel_trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(
                        receiver_buffer_index);
                if constexpr (can_defer_local_atomic_flush<enable_first_level_ack>()) {
                    // A deferred flushed atomic is pending (encoding and rules: can_defer_local_atomic_flush). Only
                    // a plain local write that issues nothing on write_at_cmd_buf may pass it: an intermesh egress
                    // forward is stateless, and its credit write would overwrite the staged atomic.
                    if (pending_atomic_flush) {
                        if (pending_atomic_flush == 1u) {
                            complete_pending_atomic_flush(receiver_channel);
                            pending_atomic_flush = 0;
                        } else {
#if defined(ARCH_BLACKHOLE) && defined(FABRIC_2D)
                            constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
                            const uint32_t trid_s = pending_atomic_flush & 0xFu;
                            const bool can_bypass =
                                hdr_w40.noc_send_type == tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE &&
                                (use_2d_single_live_stateful_noc<DOWNSTREAM_EDM_SIZE>() ||
                                 action == Routing2DCodec::ACTION_LOCAL_DELIVER) &&
                                !intermesh_egress && ((trid_s - trid) & 0xFu) > 1u;
                            if (!can_bypass) {
                                drain_pending_atomic_flush(receiver_channel, pending_atomic_flush);
                            } else if (pending_atomic_flush < 0x100u) {
                                // First write past the atomic: nothing has been issued since its packet, so every
                                // outstanding write is older. Read global outstanding first, then trid_s and
                                // trid_p = (trid_s - 1) & 0xF. Since counts only fall, g_out == s_out + p_out
                                // proves no write is outstanding on any other trid (regardless of ack order).
                                const uint32_t trid_p = (trid_s - 1u) & 0xFu;
                                const uint32_t acked_target = noc_nonposted_writes_acked[noc];
                                uint32_t g_out;
                                uint32_t s_out;
                                uint32_t p_out;
                                do {
                                    const uint32_t ack_rec = NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED);
                                    s_out = NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(trid_s));
                                    p_out = NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(trid_p));
                                    std::atomic_signal_fence(std::memory_order_seq_cst);
                                    g_out = acked_target - ack_rec;
                                } while (g_out != s_out + p_out);
                                if ((s_out | p_out) == 0) {
                                    noc_semaphore_inc_with_state<true, /*skip_cmdbuf_chk=*/true>(noc);
                                    pending_atomic_flush = 0;
                                } else {
                                    const uint32_t wait_a = s_out != 0 ? trid_s : trid_p;
                                    const uint32_t wait_b = p_out != 0 ? trid_p : trid_s;
                                    pending_atomic_flush = 0x100u | (wait_b << 4) | wait_a;
                                }
                            } else {
                                const uint32_t trid_b = (pending_atomic_flush >> 4) & 0xFu;
                                if (ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, trid_s)) {
                                    if (trid_b == trid_s ||
                                        ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, trid_b)) {
                                        noc_semaphore_inc_with_state<true, /*skip_cmdbuf_chk=*/true>(noc);
                                        pending_atomic_flush = 0;
                                    } else {
                                        pending_atomic_flush = 0x100u | (trid_b << 4) | trid_b;
                                    }
                                }
                            }
#endif
                        }
                    }
                }
                if constexpr (is_2d_fabric) {
#if defined(FABRIC_2D)
                    if (intermesh_egress) {
                        // The exit chip's only output is the INTERMESH egress, sent as-is.
                        forward_payload_to_downstream_edm<enable_deadlock_avoidance, false>(
                            packet_header,
                            packet_header->payload_size_bytes,
                            cached_routing_fields,
                            downstream_edm_interfaces[intermesh_egress_index],
                            trid);
                        record_2d_forwarded_slots<receiver_channel>(1u << intermesh_egress_index);
                    } else {
                        // Same action byte as admission. Local delivery stays outside the scan.
                        const uint8_t forwarded_slot_mask = forward_2d_dispatch(
                            action, packet_header, cached_routing_fields, downstream_edm_interfaces, trid);
                        record_2d_forwarded_slots<receiver_channel>(forwarded_slot_mask);
                        if (local_deliver) {
                            forward_to_local_destination<receiver_channel>(
                                local_relay_interface,
                                packet_header,
                                hdr_w40.payload_size_bytes,
                                hdr_w40.noc_send_type,
                                trid,
                                can_defer_local_atomic_flush<enable_first_level_ack>() ? &pending_atomic_flush
                                                                                       : nullptr);
                        }
                    }
#endif
                } else {
#ifndef FABRIC_2D
                    receiver_forward_packet<receiver_channel>(
                        packet_header, cached_routing_fields, downstream_edm_interfaces[0], trid);
#endif
                }
            }
            wr_sent_counter.increment();
            if constexpr (!enable_first_level_ack) {
                increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);
            }
        }
    }

    if constexpr (can_defer_local_atomic_flush<enable_first_level_ack>()) {
        if (!progress && pending_atomic_flush) {
            drain_pending_atomic_flush(receiver_channel, pending_atomic_flush);
        }
    }

    // Close the code profiling timer
    receiver_forward_timer.close();

    if constexpr (!fuse_receiver_flush_and_completion_ptr) {
        auto& wr_flush_counter = receiver_channel_pointers.wr_flush_counter;
        bool unflushed_writes = !wr_flush_counter.is_caught_up_to(wr_sent_counter);
        if (unflushed_writes) {
            auto receiver_buffer_index = wr_flush_counter.get_buffer_index();
            bool next_trid_flushed = receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
            if (next_trid_flushed) {
                wr_flush_counter.increment();
                receiver_channel_trid_tracker.clear_trid_at_buffer_slot(receiver_buffer_index);
            }
        }

        auto& completion_counter = receiver_channel_pointers.completion_counter;
        bool unsent_completions = !completion_counter.is_caught_up_to(completion_counter, wr_flush_counter);
        if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
            unsent_completions = unsent_completions && !internal_::eth_txq_is_busy(receiver_txq_id);
        }
        if (unsent_completions) {
            auto receiver_buffer_index = wr_flush_counter.get_buffer_index();
            receiver_send_completion_ack<ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK>(
                receiver_channel_response_credit_sender,
                receiver_channel_pointers.get_src_chan_id(receiver_buffer_index));
            completion_counter.increment();
        }
    } else if constexpr (rx_batch_completion && !enable_first_level_ack) {
        // Batched completions: one credit frame for up to batch_target written slots. While packets keep arriving,
        // send once more than a batch is waiting. When idle, send a full batch at once, and a partial one after 4
        // consecutive idle polls in 2D (at once in 1D).
        constexpr uint32_t batch_target = RECEIVER_NUM_BUFFERS_ARRAY[receiver_channel] >= 8 ? 4 : 1;
        auto& completion_counter = receiver_channel_pointers.completion_counter;
        const uint32_t uncompleted = completion_counter.distance_behind(wr_sent_counter);
#if defined(FABRIC_2D)
        bool should_check_completion;
        if (progress) {
            rx_idle_steps = 0;
            should_check_completion = uncompleted > batch_target;
        } else {
            should_check_completion = uncompleted >= batch_target || (uncompleted != 0 && ++rx_idle_steps >= 4);
        }
#else
        const bool should_check_completion = uncompleted > batch_target || (!progress && uncompleted != 0);
#endif
        if (should_check_completion) {
            const uint32_t batch = uncompleted > batch_target ? batch_target : uncompleted;
            const auto base_idx = completion_counter.get_buffer_index();
            const auto last_idx = tt::tt_fabric::BufferIndex{
                tt::tt_fabric::wrap_increment_n<RECEIVER_NUM_BUFFERS_ARRAY[receiver_channel]>(
                    base_idx.get(), batch - 1)};
            // Checks the batch's last slot and, on an idle pass, its first; the slots in between are not checked.
            bool can_send_completion = receiver_channel_trid_tracker.transaction_flushed(last_idx);
            if (!progress && batch > 1) {
                can_send_completion =
                    can_send_completion && receiver_channel_trid_tracker.transaction_flushed(base_idx);
            }
            if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
                can_send_completion = can_send_completion && !internal_::eth_txq_is_busy(receiver_txq_id);
            }
            if (can_send_completion) {
#if defined(FABRIC_2D)
                rx_idle_steps = 0;
#endif
                if constexpr (skip_src_ch_id_update) {
                    receiver_send_completion_ack<ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK>(
                        receiver_channel_response_credit_sender, receiver_channel_pointers.get_src_chan_id(), batch);
                } else {
                    auto idx = base_idx;
                    for (uint32_t k = 0; k < batch; k++) {
                        receiver_channel_response_credit_sender.accumulate_completion_credit(
                            receiver_channel_pointers.get_src_chan_id(idx), 1);
                        receiver_channel_trid_tracker.clear_trid_at_buffer_slot(idx);
                        idx = tt::tt_fabric::BufferIndex{
                            tt::tt_fabric::wrap_increment<RECEIVER_NUM_BUFFERS_ARRAY[receiver_channel]>(idx.get())};
                    }
                    if constexpr (ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
                        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
                        }
                    }
                    receiver_channel_response_credit_sender.flush_completion_credits();
                }
                completion_counter.increment_n(batch);
            }
        }
    } else {
        auto& completion_counter = receiver_channel_pointers.completion_counter;
        // Currently unclear if it's better to loop here or not...
        bool unflushed_writes = !completion_counter.is_caught_up_to(wr_sent_counter);
        auto receiver_buffer_index = completion_counter.get_buffer_index();
        bool next_trid_flushed = receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
        bool can_send_completion = unflushed_writes && next_trid_flushed;
        if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
            can_send_completion = can_send_completion && !internal_::eth_txq_is_busy(receiver_txq_id);
        }
        if (can_send_completion) {
            uint8_t src_ch_id;
            if constexpr (skip_src_ch_id_update) {
                src_ch_id = receiver_channel_pointers.get_src_chan_id();
            } else {
                src_ch_id = receiver_channel_pointers.get_src_chan_id(receiver_buffer_index);
            }
            receiver_send_completion_ack<ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK>(
                receiver_channel_response_credit_sender, src_ch_id);
            receiver_channel_trid_tracker.clear_trid_at_buffer_slot(receiver_buffer_index);
            completion_counter.increment();
        }
    }
    return progress;
}

template <
    uint8_t receiver_channel,
    bool enable_first_level_ack,
    size_t DOWNSTREAM_EDM_SIZE,
    typename DownstreamSenderT,
    typename LocalRelayInterfaceT,
    typename EthReceiverChannels,
    typename WriteTridTracker,
    typename ReceiverChannelPointersT,
    typename LocalTelemetryT>
FORCE_INLINE bool run_receiver_channel_step(
    EthReceiverChannels& local_receiver_channels,
    std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    LocalRelayInterfaceT& local_relay_interface,
    ReceiverChannelPointersT& receiver_channel_pointers,
    WriteTridTracker& receiver_channel_trid_tracker,
    std::array<uint8_t, num_eth_ports>& port_direction_table,
    ReceiverChannelResponseCreditSenders<NUM_RECEIVER_CHANNELS>& receiver_channel_response_credit_senders,
    const tt::tt_fabric::routing_l1_info_t& routing_table,
    uint16_t my_mesh_id,
    uint8_t my_mesh_coord_y,
    uint8_t my_mesh_coord_x,
    LocalTelemetryT& local_fabric_telemetry,
    uint32_t& rx_idle_steps,
    uint32_t& pending_atomic_flush) {
    if constexpr (is_receiver_channel_serviced[receiver_channel]) {
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        const bool progress = run_receiver_channel_step_impl<
            receiver_channel,
            to_receiver_packets_sent_streams[receiver_channel],
            enable_first_level_ack,
            DOWNSTREAM_EDM_SIZE,
            WriteTridTracker,
            decltype(local_receiver_channels.template get<receiver_channel>()),
            ReceiverChannelPointersT,
            DownstreamSenderT,
            LocalRelayInterfaceT>(
            local_receiver_channels.template get<receiver_channel>(),
            downstream_edm_interfaces,
            local_relay_interface,
            receiver_channel_pointers,
            receiver_channel_trid_tracker,
            port_direction_table,
            receiver_channel_response_credit_senders.template get<receiver_channel>(),
            routing_table,
            my_mesh_id,
            my_mesh_coord_y,
            my_mesh_coord_x,
            local_fabric_telemetry,
            rx_idle_steps,
            pending_atomic_flush);
        return progress;
    }
    return false;
}

template <typename OutboundReceiverChannelPointers, typename RemoteEthReceiverChannels>
FORCE_INLINE void configure_outbound_to_receiver_channel_pointers(
    OutboundReceiverChannelPointers& outbound_to_receiver_channel_pointers,
    RemoteEthReceiverChannels& remote_receiver_channels) {
    static_assert(OutboundReceiverChannelPointers::N == RemoteEthReceiverChannels::num_channels);

    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((outbound_to_receiver_channel_pointers.template get<Is>().init(
             reinterpret_cast<uint32_t>(remote_receiver_channels.template get<Is>().channel_base_address()),
             remote_receiver_channels.template get<Is>().get_max_eth_payload_size())),
         ...);
    }(std::make_index_sequence<OutboundReceiverChannelPointers::N>{});
}

void run_retrain_step(
    tt_l1_ptr RouterStateManager* state_manager_l1, volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    // Placeholder
    state_manager_l1->state = RouterState::RETRAINING;
    run_routing_without_noc_sync();
    while (reinterpret_cast<tt_l1_ptr RouterStateManager*>(state_manager_l1)->command == RouterCommand::RETRAIN &&
           !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
    }
    ASSERT(reinterpret_cast<tt_l1_ptr RouterStateManager*>(state_manager_l1)->command == RouterCommand::PAUSE);
}

void run_drain_step(
    tt_l1_ptr RouterStateManager* state_manager_l1, volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    state_manager_l1->state = RouterState::DRAINING;
    ASSERT(false);  // not implemented
    while (reinterpret_cast<tt_l1_ptr RouterStateManager*>(state_manager_l1)->command == RouterCommand::DRAIN &&
           !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
    }
    ASSERT(reinterpret_cast<tt_l1_ptr RouterStateManager*>(state_manager_l1)->command == RouterCommand::PAUSE);
}

__attribute__((noinline)) void execute_pause_command(
    tt_l1_ptr RouterStateManager* state_manager_l1, volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    if constexpr (MY_ERISC_ID == 0) {
        bool keep_running_pause = true;
        while (keep_running_pause &&
               !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
            state_manager_l1->state = RouterState::PAUSED;
            switch (reinterpret_cast<tt_l1_ptr RouterStateManager*>(state_manager_l1)->command) {
                case RouterCommand::RUN: keep_running_pause = false; break;
                case RouterCommand::PAUSE: run_routing_without_noc_sync(); break;
                case RouterCommand::DRAIN: run_drain_step(state_manager_l1, termination_signal_ptr); break;
                case RouterCommand::RETRAIN: run_retrain_step(state_manager_l1, termination_signal_ptr); break;
                default: ASSERT(false);
            }
        }
    }
}

template <bool RISC_CPU_DATA_CACHE_ENABLED>
FORCE_INLINE bool continue_running_main_run_loop(
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr, tt_l1_ptr RouterStateManager* state_manager_l1) {
    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        if constexpr (!IS_RETRAIN_SYNC_MASTER()) {
            return !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr);
        } else {
            return !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr) &&
                   !state_manager_l1->is_non_run_command_pending();
        }
    } else {
        return !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr) &&
               !state_manager_l1->is_non_run_command_pending();
    }
}

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_speedy_path.hpp"

// The generic loop: every build that sender_erisc and receiver_erisc do not cover (see the top of this file). Each
// outer iteration checks for termination (and, on the context-switch master, pending router commands), runs
// iterations_between_ctx_switch_and_teardown_checks passes over the channels, then updates telemetry, the heartbeat
// and the idle context switch. A channel step compiles to nothing on an ERISC that does not service its channel.
// STABILITY_PROBE_DECL
template <
    size_t NUM_RECEIVER_CHANNELS,
    typename DownstreamSenderVC0T,
    typename DownstreamSenderVC1T,
    typename LocalRelayInterfaceT,
    typename EthSenderChannels,
    typename EthReceiverChannels,
    typename RemoteEthReceiverChannels,
    typename EdmChannelWorkerIFs,
    typename TransactionIdTrackerCH0
#if defined(FABRIC_2D_VC1_ACTIVE)
    ,
    typename TransactionIdTrackerCH1
#endif
#if defined(FABRIC_2D_VC2_SERVICED)
    ,
    typename TransactionIdTrackerCH2
#endif
    >
FORCE_INLINE void run_fabric_edm_main_loop(
    EthReceiverChannels& local_receiver_channels,
    EthSenderChannels& local_sender_channels,
    EdmChannelWorkerIFs& local_sender_channel_worker_interfaces,
    std::array<DownstreamSenderVC0T, VC0_DOWNSTREAM_EDM_SIZE>& downstream_edm_noc_interfaces_vc0,
    std::array<DownstreamSenderVC1T, VC1_DOWNSTREAM_EDM_SIZE>& downstream_edm_noc_interfaces_vc1,
    LocalRelayInterfaceT& local_relay_interface,
    RemoteEthReceiverChannels& remote_receiver_channels,
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr,
    TransactionIdTrackerCH0& receiver_channel_0_trid_tracker,
#if defined(FABRIC_2D_VC1_ACTIVE)
    TransactionIdTrackerCH1& receiver_channel_1_trid_tracker,
#endif
#if defined(FABRIC_2D_VC2_SERVICED)
    TransactionIdTrackerCH2& receiver_channel_2_trid_tracker,
#endif
    std::array<uint8_t, num_eth_ports>& port_direction_table,
    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids) {
    size_t did_nothing_count = 0;
    using FabricTelemetryT = FabricTelemetry;
    FabricTelemetryT local_fabric_telemetry{};
    auto fabric_telemetry = reinterpret_cast<volatile FabricTelemetryT*>(MEM_AERISC_FABRIC_TELEMETRY_BASE);

    // Read the routing table in place
    const auto& routing_table =
        *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    auto* state_manager_l1 = const_cast<tt_l1_ptr RouterStateManager*>(&routing_table.state_manager);

    // This router's coordinates and mesh id, read once (the routing table itself is read in place).
    const uint8_t my_mesh_coord_y = routing_table.my_mesh_coord_y;
    const uint8_t my_mesh_coord_x = routing_table.my_mesh_coord_x;
    const uint16_t my_mesh_id = routing_table.my_mesh_id;

    // May want to promote to part of the handshake but for now we just initialize in this standalone way
    // TODO: flatten all of these arrays into a single object (one array lookup) OR
    //       (probably better) pack most of these into single words (e.g. we could hold a read, write, and ackptr in a
    //       single word) this way - especially if power of 2 wraps, we can handle both channels literally at once with
    //       math ops on single individual words (or half words)
    auto outbound_to_receiver_channel_pointers =
        ChannelPointersTuple<OutboundReceiverChannelPointers, REMOTE_RECEIVER_NUM_BUFFERS_ARRAY>::make();
    configure_outbound_to_receiver_channel_pointers(outbound_to_receiver_channel_pointers, remote_receiver_channels);

    auto receiver_channel_pointers = ChannelPointersTuple<ReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>::make();

    // Each channel's pointers are copied out of the tuples into their own locals, as on main, where using them
    // through the tuple regressed the RingAsLinear test (an allocation effect, not a semantic one).
    auto outbound_to_receiver_channel_pointer_ch0 =
        outbound_to_receiver_channel_pointers.template get<VC0_RECEIVER_CHANNEL>();

    auto receiver_channel_pointers_ch0 = receiver_channel_pointers.template get<0>();
    receiver_channel_pointers_ch0.reset();
    uint32_t rx_idle_steps_ch0 = 0;
    uint32_t pending_atomic_flush_ch0 = 0;

#if defined(FABRIC_2D_VC1_SERVICED)
    // VC1 receiver channel pointer for inter-mesh routing. Gated on _SERVICED (not _ACTIVE) because
    // these locals are only consumed inside the FABRIC_2D_VC1_SERVICED block further down; on routers
    // that have VC1 channels (ACTIVE) but do not service them (e.g. inter-mesh routers without
    // pass-through), defining them here trips -Wunused-but-set-variable in the kernel JIT compile.
    auto outbound_to_receiver_channel_pointer_ch1 =
        outbound_to_receiver_channel_pointers.template get<VC1_RECEIVER_CHANNEL>();

    auto receiver_channel_pointers_ch1 = receiver_channel_pointers.template get<1>();
    receiver_channel_pointers_ch1.reset();
    uint32_t rx_idle_steps_ch1 = 0;
    uint32_t pending_atomic_flush_ch1 = 0;
#endif

#if defined(FABRIC_2D_VC2_SERVICED)
    auto outbound_to_receiver_channel_pointer_ch2 =
        outbound_to_receiver_channel_pointers.template get<VC2_RECEIVER_CHANNEL>();

    auto receiver_channel_pointers_ch2 = receiver_channel_pointers.template get<VC2_RECEIVER_CHANNEL>();
    receiver_channel_pointers_ch2.reset();
#endif

    if constexpr (skip_src_ch_id_update) {
        receiver_channel_pointers_ch0.set_src_chan_id(BufferIndex{0}, remote_worker_sender_channel);
    }

    std::array<bool, NUM_SENDER_CHANNELS> channel_connection_established =
        initialize_array<NUM_SENDER_CHANNELS, bool, false>();

    PerfTelemetryRecorder inner_loop_perf_telemetry_collector = build_perf_telemetry_recorder<perf_telemetry_mode>();
    auto local_perf_telemetry_buffer =
        build_perf_telemetry_buffer(reinterpret_cast<uint32_t*>(perf_telemetry_buffer_addr));

    auto receiver_channel_response_credit_senders =
        init_receiver_channel_response_credit_senders<NUM_RECEIVER_CHANNELS>();
    auto sender_channel_from_receiver_credits =
        init_sender_channel_from_receiver_credits_flow_controllers<NUM_SENDER_CHANNELS>();

    static ActualSpeedySenderState<super_speedy_mode> persistent_speedy_sender_state;
    static ActualSpeedyReceiverState<super_speedy_mode, 0> persistent_speedy_receiver_state;
#if defined(FABRIC_2D_VC2_SERVICED)
    static ActualSpeedySenderState<true> persistent_speedy_sender_state_vc2;
    static ActualSpeedyReceiverState<true, 2> persistent_speedy_receiver_state_vc2;
#endif

    uint16_t fabric_heartbeat_counter = 0;
    volatile uint32_t* fabric_heartbeat_ptr = reinterpret_cast<volatile uint32_t*>(FABRIC_KERNEL_HEARTBEAT_ADDR);

    auto execute_main_loop = [&]() __attribute__((always_inline)) {
        ActualSpeedySenderState<super_speedy_mode> local_speedy_sender_state;
        ActualSpeedyReceiverState<super_speedy_mode, 0> local_speedy_receiver_state;
#if defined(FABRIC_2D_VC2_SERVICED)
        ActualSpeedySenderState<true> local_speedy_sender_state_vc2;
        ActualSpeedyReceiverState<true, 2> local_speedy_receiver_state_vc2;
#endif
        speedy_state_copy_in<super_speedy_mode, 0>(
            local_speedy_sender_state,
            local_speedy_receiver_state,
            persistent_speedy_sender_state,
            persistent_speedy_receiver_state);
#if defined(FABRIC_2D_VC2_SERVICED)
        speedy_state_copy_in<true, 2>(
            local_speedy_sender_state_vc2,
            local_speedy_receiver_state_vc2,
            persistent_speedy_sender_state_vc2,
            persistent_speedy_receiver_state_vc2);
#endif
        // STABILITY_PROBE_PRELOOP
        while (continue_running_main_run_loop<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr, state_manager_l1)) {
            did_something = false;

            uint32_t tx_progress = 0;
            uint32_t rx_progress = 0;

            if constexpr (any_sender_channel_serviced) {
                open_perf_recording_window(inner_loop_perf_telemetry_collector);
            }

            uint64_t loop_start_cycles;
            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                loop_start_cycles = get_timestamp();
            }
#if defined(FABRIC_2D_VC2_SERVICED)
            if constexpr (is_sender_channel_serviced[VC2_LOCAL_CHANNEL_START]) {
                auto check_connection_status =
                    !channel_connection_established[VC2_LOCAL_CHANNEL_START] ||
                    local_sender_channel_worker_interfaces.template get<VC2_LOCAL_CHANNEL_START>()
                        .has_worker_teardown_request();
                if (check_connection_status) {
                    check_worker_connections<MY_ETH_CHANNEL, ENABLE_RISC_CPU_DATA_CACHE>(
                        local_sender_channel_worker_interfaces.template get<VC2_LOCAL_CHANNEL_START>(),
                        channel_connection_established[VC2_LOCAL_CHANNEL_START],
                        local_sender_channel_free_slots_stream_ids[VC2_FABRIC_POSITION_START]);
                }
            }
#endif
            // iterations_between_ctx_switch_and_teardown_checks (a CT arg, 32) is the number of passes between exits
            // to check for termination and context switch. Removing these checks from the inner loop can drastically
            // improve performance. The value of 32 was chosen somewhat empirically and then raised up slightly.
            for (size_t i = 0; i < iterations_between_ctx_switch_and_teardown_checks; i++) {
                // STABILITY_PROBE_LOOP
                if constexpr (super_speedy_mode) {
                    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();

                    if constexpr (is_sender_channel_serviced[0]) {
                        tx_progress |= run_sender_channel_step_speedy<
                            0,
                            to_receiver_packets_sent_streams[VC0_RECEIVER_CHANNEL],
                            SENDER_CREDIT_AMORTIZATION_FREQUENCY,
                            true>(
                            local_sender_channels.template get<0>(),
                            local_sender_channel_worker_interfaces.template get<0>(),
                            outbound_to_receiver_channel_pointer_ch0,
                            remote_receiver_channels.template get<VC0_RECEIVER_CHANNEL>(),
                            channel_connection_established[0],
                            local_sender_channel_free_slots_stream_ids[0],
                            sender_channel_from_receiver_credits.template get<0>(),
                            inner_loop_perf_telemetry_collector,
                            local_fabric_telemetry,
                            local_speedy_sender_state);
                    }

                    if constexpr (is_receiver_channel_serviced[0]) {
                        rx_progress |= run_receiver_channel_step_speedy<
                            0,
                            0,
                            to_receiver_packets_sent_streams[0],
                            RECEIVER_CREDIT_AMORTIZATION_FREQUENCY,
                            decltype(receiver_channel_0_trid_tracker)>(
                            local_receiver_channels.template get<0>(),
                            local_relay_interface,
                            receiver_channel_pointers_ch0,
                            receiver_channel_0_trid_tracker,
                            port_direction_table,
                            receiver_channel_response_credit_senders.template get<0>(),
                            routing_table,
                            local_fabric_telemetry,
                            local_speedy_receiver_state);
                    }
                } else {
                    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
                    uint32_t free_slots_raw[4] = {0, 0, 0, 0};
                    if constexpr (any_sender_channel_serviced) {
                        read_grouped_free_slots(free_slots_raw);
                    }
                    tx_progress |= run_sender_channel_step<VC0_RECEIVER_CHANNEL, 0, ENABLE_FIRST_LEVEL_ACK_VC0>(
                        local_sender_channels,
                        local_sender_channel_worker_interfaces,
                        outbound_to_receiver_channel_pointer_ch0,
                        channel_connection_established,
                        local_sender_channel_free_slots_stream_ids,
                        sender_channel_from_receiver_credits,
                        inner_loop_perf_telemetry_collector,
                        local_fabric_telemetry,
                        free_slots_raw[0]);
#if defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)
                    rx_progress |= run_receiver_channel_step<
                        0,
                        ENABLE_FIRST_LEVEL_ACK_VC0,
                        VC1_DOWNSTREAM_EDM_SIZE,
                        DownstreamSenderVC1T,
                        decltype(local_relay_interface)>(
                        local_receiver_channels,
                        downstream_edm_noc_interfaces_vc1,
                        local_relay_interface,
                        receiver_channel_pointers_ch0,
                        receiver_channel_0_trid_tracker,
                        port_direction_table,
                        receiver_channel_response_credit_senders,
                        routing_table,
                        my_mesh_id,
                        my_mesh_coord_y,
                        my_mesh_coord_x,
                        local_fabric_telemetry,
                        rx_idle_steps_ch0,
                        pending_atomic_flush_ch0);
#else
                    rx_progress |= run_receiver_channel_step<
                        0,
                        ENABLE_FIRST_LEVEL_ACK_VC0,
                        VC0_DOWNSTREAM_EDM_SIZE,
                        DownstreamSenderVC0T,
                        decltype(local_relay_interface)>(
                        local_receiver_channels,
                        downstream_edm_noc_interfaces_vc0,
                        local_relay_interface,
                        receiver_channel_pointers_ch0,
                        receiver_channel_0_trid_tracker,
                        port_direction_table,
                        receiver_channel_response_credit_senders,
                        routing_table,
                        my_mesh_id,
                        my_mesh_coord_y,
                        my_mesh_coord_x,
                        local_fabric_telemetry,
                        rx_idle_steps_ch0,
                        pending_atomic_flush_ch0);
#endif
                    tx_progress |= run_sender_channel_step<VC0_RECEIVER_CHANNEL, 1, ENABLE_FIRST_LEVEL_ACK_VC0>(
                        local_sender_channels,
                        local_sender_channel_worker_interfaces,
                        outbound_to_receiver_channel_pointer_ch0,
                        channel_connection_established,
                        local_sender_channel_free_slots_stream_ids,
                        sender_channel_from_receiver_credits,
                        inner_loop_perf_telemetry_collector,
                        local_fabric_telemetry,
                        free_slots_raw[1]);
                    if constexpr (is_2d_fabric) {
                        tx_progress |= run_sender_channel_step<VC0_RECEIVER_CHANNEL, 2, ENABLE_FIRST_LEVEL_ACK_VC0>(
                            local_sender_channels,
                            local_sender_channel_worker_interfaces,
                            outbound_to_receiver_channel_pointer_ch0,
                            channel_connection_established,
                            local_sender_channel_free_slots_stream_ids,
                            sender_channel_from_receiver_credits,
                            inner_loop_perf_telemetry_collector,
                            local_fabric_telemetry,
                            free_slots_raw[2]);
                        tx_progress |= run_sender_channel_step<VC0_RECEIVER_CHANNEL, 3, ENABLE_FIRST_LEVEL_ACK_VC0>(
                            local_sender_channels,
                            local_sender_channel_worker_interfaces,
                            outbound_to_receiver_channel_pointer_ch0,
                            channel_connection_established,
                            local_sender_channel_free_slots_stream_ids,
                            sender_channel_from_receiver_credits,
                            inner_loop_perf_telemetry_collector,
                            local_fabric_telemetry,
                            free_slots_raw[3]);
                    }
                    if constexpr (is_2d_fabric && ACTUAL_VC0_SENDER_CHANNELS > 4) {
                        tx_progress |= run_sender_channel_step<VC0_RECEIVER_CHANNEL, 4, ENABLE_FIRST_LEVEL_ACK_VC0>(
                            local_sender_channels,
                            local_sender_channel_worker_interfaces,
                            outbound_to_receiver_channel_pointer_ch0,
                            channel_connection_established,
                            local_sender_channel_free_slots_stream_ids,
                            sender_channel_from_receiver_credits,
                            inner_loop_perf_telemetry_collector,
                            local_fabric_telemetry);
                    }
                }
                if constexpr (is_2d_fabric) {
#if defined(FABRIC_2D_VC1_SERVICED)
                    tx_progress |= run_sender_channel_step<
                        VC1_RECEIVER_CHANNEL,
                        ACTUAL_VC0_SENDER_CHANNELS,
                        ENABLE_FIRST_LEVEL_ACK_VC1>(
                        local_sender_channels,
                        local_sender_channel_worker_interfaces,
                        outbound_to_receiver_channel_pointer_ch1,
                        channel_connection_established,
                        local_sender_channel_free_slots_stream_ids,
                        sender_channel_from_receiver_credits,
                        inner_loop_perf_telemetry_collector,
                        local_fabric_telemetry);
                    tx_progress |= run_sender_channel_step<
                        VC1_RECEIVER_CHANNEL,
                        ACTUAL_VC0_SENDER_CHANNELS + 1,
                        ENABLE_FIRST_LEVEL_ACK_VC1>(
                        local_sender_channels,
                        local_sender_channel_worker_interfaces,
                        outbound_to_receiver_channel_pointer_ch1,
                        channel_connection_established,
                        local_sender_channel_free_slots_stream_ids,
                        sender_channel_from_receiver_credits,
                        inner_loop_perf_telemetry_collector,
                        local_fabric_telemetry);
                    tx_progress |= run_sender_channel_step<
                        VC1_RECEIVER_CHANNEL,
                        ACTUAL_VC0_SENDER_CHANNELS + 2,
                        ENABLE_FIRST_LEVEL_ACK_VC1>(
                        local_sender_channels,
                        local_sender_channel_worker_interfaces,
                        outbound_to_receiver_channel_pointer_ch1,
                        channel_connection_established,
                        local_sender_channel_free_slots_stream_ids,
                        sender_channel_from_receiver_credits,
                        inner_loop_perf_telemetry_collector,
                        local_fabric_telemetry);
                    if constexpr (ACTUAL_VC1_SENDER_CHANNELS > 3) {
                        tx_progress |= run_sender_channel_step<
                            VC1_RECEIVER_CHANNEL,
                            ACTUAL_VC0_SENDER_CHANNELS + 3,
                            ENABLE_FIRST_LEVEL_ACK_VC1>(
                            local_sender_channels,
                            local_sender_channel_worker_interfaces,
                            outbound_to_receiver_channel_pointer_ch1,
                            channel_connection_established,
                            local_sender_channel_free_slots_stream_ids,
                            sender_channel_from_receiver_credits,
                            inner_loop_perf_telemetry_collector,
                            local_fabric_telemetry);
                    }
                    rx_progress |= run_receiver_channel_step<
                        1,
                        ENABLE_FIRST_LEVEL_ACK_VC1,
                        VC1_DOWNSTREAM_EDM_SIZE,
                        DownstreamSenderVC1T,
                        decltype(local_relay_interface)>(
                        local_receiver_channels,
                        downstream_edm_noc_interfaces_vc1,
                        local_relay_interface,
                        receiver_channel_pointers_ch1,
                        receiver_channel_1_trid_tracker,
                        port_direction_table,
                        receiver_channel_response_credit_senders,
                        routing_table,
                        my_mesh_id,
                        my_mesh_coord_y,
                        my_mesh_coord_x,
                        local_fabric_telemetry,
                        rx_idle_steps_ch1,
                        pending_atomic_flush_ch1);
#endif

#if defined(FABRIC_2D_VC2_SERVICED)
                    if constexpr (is_sender_channel_serviced[VC2_LOCAL_CHANNEL_START]) {
                        constexpr size_t SENDER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL_VC2 = 1;
                        tx_progress |= run_sender_channel_step_speedy<
                            VC2_LOCAL_CHANNEL_START,
                            to_receiver_packets_sent_streams[VC2_RECEIVER_CHANNEL],
                            SENDER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL_VC2,
                            false>(
                            local_sender_channels.template get<VC2_LOCAL_CHANNEL_START>(),
                            local_sender_channel_worker_interfaces.template get<VC2_LOCAL_CHANNEL_START>(),
                            outbound_to_receiver_channel_pointer_ch2,
                            remote_receiver_channels.template get<VC2_RECEIVER_CHANNEL>(),
                            channel_connection_established[VC2_LOCAL_CHANNEL_START],
                            local_sender_channel_free_slots_stream_ids[VC2_FABRIC_POSITION_START],
                            sender_channel_from_receiver_credits.template get<VC2_LOCAL_CHANNEL_START>(),
                            inner_loop_perf_telemetry_collector,
                            local_fabric_telemetry,
                            local_speedy_sender_state_vc2);
                    }
                    if constexpr (is_receiver_channel_serviced[VC2_RECEIVER_CHANNEL]) {
                        constexpr size_t RECEIVER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL_VC2 = 1;
                        rx_progress |= run_receiver_channel_step_speedy<
                            VC2_RECEIVER_CHANNEL,
                            VC2_LOCAL_CHANNEL_START,
                            to_receiver_packets_sent_streams[VC2_RECEIVER_CHANNEL],
                            RECEIVER_CREDIT_AMORTIZATION_FREQUENCY_LOCAL_VC2,
                            decltype(receiver_channel_2_trid_tracker)>(
                            local_receiver_channels.template get<VC2_RECEIVER_CHANNEL>(),
                            local_relay_interface,
                            receiver_channel_pointers_ch2,
                            receiver_channel_2_trid_tracker,
                            port_direction_table,
                            receiver_channel_response_credit_senders.template get<VC2_RECEIVER_CHANNEL>(),
                            routing_table,
                            local_fabric_telemetry,
                            local_speedy_receiver_state_vc2);
                    }
#endif
                }
            }
            if constexpr (
                is_receiver_channel_serviced[0] && can_defer_local_atomic_flush<ENABLE_FIRST_LEVEL_ACK_VC0>()) {
                if (pending_atomic_flush_ch0) {
                    drain_pending_atomic_flush(0, pending_atomic_flush_ch0);
                }
            }

            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                uint64_t loop_end_cycles = get_timestamp();
                uint64_t loop_delta_cycles = loop_end_cycles - loop_start_cycles;
                update_bw_cycles(loop_delta_cycles, tx_progress, rx_progress, local_fabric_telemetry);
            }
            // Compute idle conditions and update heartbeats in one helper
            if constexpr (FABRIC_TELEMETRY_ANY_DYNAMIC_STAT) {
                update_telemetry(
                    local_sender_channel_free_slots_stream_ids,
                    tx_progress,
                    rx_progress,
                    local_fabric_telemetry,
                    fabric_telemetry);
            }

            if ((++fabric_heartbeat_counter & 0x3F) == 0) {
                *fabric_heartbeat_ptr = 0xDCBA0000 | fabric_heartbeat_counter;
            }

            if constexpr (enable_context_switch) {
                // shouldn't do noc counter sync since we are not incrementing them
                if constexpr (IDLE_CONTEXT_SWITCHING) {
                    if (did_something) {
                        did_nothing_count = 0;
                    } else {
                        if (did_nothing_count++ > SWITCH_INTERVAL) {
                            did_nothing_count = 0;
                            run_coordinated_context_switch_to_base_firmware(termination_signal_ptr);
                        }
                    }
                } else {
                    if (did_nothing_count++ > SWITCH_INTERVAL) {
                        did_nothing_count = 0;
                        run_coordinated_context_switch_to_base_firmware(termination_signal_ptr);
                    }
                }
            } else {
                run_routing_without_noc_sync_coordinated_as_non_master(termination_signal_ptr);
            }

            if constexpr (any_sender_channel_serviced) {
                close_perf_recording_window(inner_loop_perf_telemetry_collector);
                if constexpr (perf_telemetry_mode != PerfTelemetryRecorderType::NONE) {
                    if (captured_an_event(inner_loop_perf_telemetry_collector) ||
                        any_sender_channels_active(local_sender_channel_free_slots_stream_ids)) {
                        write_perf_recording_window_results(
                            inner_loop_perf_telemetry_collector, local_perf_telemetry_buffer);
                    }
                }
            }
        }

        speedy_state_copy_out<super_speedy_mode, 0>(
            persistent_speedy_sender_state,
            persistent_speedy_receiver_state,
            local_speedy_sender_state,
            local_speedy_receiver_state);
#if defined(FABRIC_2D_VC2_SERVICED)
        speedy_state_copy_out<true, 2>(
            persistent_speedy_sender_state_vc2,
            persistent_speedy_receiver_state_vc2,
            local_speedy_sender_state_vc2,
            local_speedy_receiver_state_vc2);
#endif
    };

    if constexpr (MY_ERISC_ID == 0) {
        while (!got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
            switch (reinterpret_cast<volatile tt_l1_ptr RouterStateManager*>(state_manager_l1)->command) {
                case RouterCommand::RUN:
                    state_manager_l1->state = RouterState::RUNNING;
                    execute_main_loop();
                    break;
                case RouterCommand::PAUSE:
                    if constexpr (NUM_ACTIVE_ERISCS > 1) {
                        coordinated_context_switch_start_as_master(termination_signal_ptr);
                    }
                    execute_pause_command(state_manager_l1, termination_signal_ptr);
                    if constexpr (NUM_ACTIVE_ERISCS > 1) {
                        coordinated_context_switch_finish_as_master(termination_signal_ptr);
                    }
                    break;
                default: ASSERT(false); break;
            }
        }
    } else {
        while (!got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
            execute_main_loop();
        }
    }
}

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_receiver_erisc.hpp"

static constexpr uint32_t CONTEXT_SWITCH_TIMEOUT_SHORT = 100000;
template <typename EdmChannelWorkerIFs>
void
#ifdef FABRIC_2D
    __attribute__((noinline))
#endif
    wait_for_static_connection_to_ready(
        EdmChannelWorkerIFs& local_sender_channel_worker_interfaces,
        std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids,
        volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    auto establish_static_connection_from_receiver_side = [&](auto& interface, size_t sender_channel_idx) {
        if (!sender_ch_wait_static_connection[sender_channel_idx]) {
            return;
        }
        uint32_t count = 0;
        while (!connect_is_requested(*interface.connection_live_semaphore)
#ifndef ARCH_WORMHOLE
               && !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)
#endif
        ) {
            router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
#ifndef ARCH_WORMHOLE
            if constexpr (MY_ERISC_ID == 0) {
                if (count++ == CONTEXT_SWITCH_TIMEOUT_SHORT) {
                    count = 0;
                    run_routing();
                }
            }
#endif
        }
        establish_edm_connection(
            interface,
            local_sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(sender_channel_idx)]);
    };
    if constexpr (multi_txq_enabled) {
        tuple_for_each_constexpr(
            local_sender_channel_worker_interfaces.channel_worker_interfaces, [&](auto& interface, auto idx) {
                if constexpr (is_sender_channel_serviced[idx]) {
                    establish_static_connection_from_receiver_side(interface, idx);
                }
            });
    } else {
        // Very slight performance regression on WH if we commonize to the above path, so we preserve this path
        // too
        tuple_for_each(
            local_sender_channel_worker_interfaces.channel_worker_interfaces,
            [&](auto& interface, size_t idx) { establish_static_connection_from_receiver_side(interface, idx); });
    }
}

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_v3_sender_erisc.hpp"

// Returns the number of starting credits for the specified sender channel `i`
// Generally, we will always start with `SENDER_NUM_BUFFERS` of credits,
// except for channels which service transient/worker connections. Those
// sender channels use counter based credit schemes so they are initialized
// to 0.
template <size_t i>
constexpr size_t get_credits_init_val() {
    if (i == 0) {
        return 0;
    }
#if defined(FABRIC_2D_VC2_SERVICED)
    if (i == VC2_LOCAL_CHANNEL_START) {
        return 0;
    }
#endif
    return SENDER_NUM_BUFFERS_ARRAY[i];
}

template <size_t I, size_t NUM_SENDER_CHANNELS, typename EdmChannelWorkerIFs>
FORCE_INLINE typename std::enable_if<(I < NUM_SENDER_CHANNELS), void>::type init_sender_channel_worker_interface(
    std::array<size_t, NUM_SENDER_CHANNELS>& local_sender_connection_live_semaphore_addresses,
    std::array<size_t, NUM_SENDER_CHANNELS>& local_sender_connection_info_addresses,
    EdmChannelWorkerIFs& local_sender_channel_worker_interfaces) {
    auto connection_live_semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(local_sender_connection_live_semaphore_addresses[I]);
    auto connection_worker_info_ptr = reinterpret_cast<volatile tt::tt_fabric::EDMChannelWorkerLocationInfo*>(
        local_sender_connection_info_addresses[I]);
    new (&local_sender_channel_worker_interfaces.template get<I>()) tt::tt_fabric::
        StaticSizedSenderChannelWorkerInterface<tt::tt_fabric::worker_handshake_noc, SENDER_NUM_BUFFERS_ARRAY[I]>(
            connection_worker_info_ptr,
            0,
            reinterpret_cast<volatile tt_l1_ptr uint32_t* const>(connection_live_semaphore_ptr),
            sender_channel_ack_cmd_buf_ids[I],
            get_credits_init_val<I>(),
            notify_worker_of_read_counter_update_src_address);
}

template <size_t I, size_t NUM_SENDER_CHANNELS, typename EdmChannelWorkerIFs>
typename std::enable_if<(I >= NUM_SENDER_CHANNELS), void>::type init_sender_channel_worker_interface(
    std::array<size_t, NUM_SENDER_CHANNELS>&, std::array<size_t, NUM_SENDER_CHANNELS>&, EdmChannelWorkerIFs&) {}

template <size_t NUM_SENDER_CHANNELS, typename EdmChannelWorkerIFs>
__attribute__((optimize("Os"))) void
#ifdef FABRIC_2D
    __attribute__((noinline))
#endif
    init_local_sender_channel_worker_interfaces(
        std::array<size_t, NUM_SENDER_CHANNELS>& local_sender_connection_live_semaphore_addresses,
        std::array<size_t, NUM_SENDER_CHANNELS>& local_sender_connection_info_addresses,
        EdmChannelWorkerIFs& local_sender_channel_worker_interfaces) {
    // manual unrol because previously, going from having this in a loop to unrolling this would
    // lead to a performance regression. Having these unrolled is needed to enable some performance optimizations
    // because setup will differ in that each will be a different type. Keeping them unrolled here let's us
    // stay safe from perf regression due to weirdness of codegen.
    init_sender_channel_worker_interface<0, NUM_SENDER_CHANNELS>(
        local_sender_connection_live_semaphore_addresses,
        local_sender_connection_info_addresses,
        local_sender_channel_worker_interfaces);
    if constexpr (NUM_SENDER_CHANNELS > 1) {
        init_sender_channel_worker_interface<1, NUM_SENDER_CHANNELS>(
            local_sender_connection_live_semaphore_addresses,
            local_sender_connection_info_addresses,
            local_sender_channel_worker_interfaces);
    }
#ifdef FABRIC_2D
    // Use compile-time loop to initialize remaining sender channels (2-6) for code size optimization
    if constexpr (NUM_SENDER_CHANNELS > 2) {
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            (([&]<size_t I>() {
                 if constexpr (NUM_SENDER_CHANNELS > I) {
                     init_sender_channel_worker_interface<I, NUM_SENDER_CHANNELS>(
                         local_sender_connection_live_semaphore_addresses,
                         local_sender_connection_info_addresses,
                         local_sender_channel_worker_interfaces);
                 }
             }.template operator()<Is + 2>()),
             ...);
        }(std::make_index_sequence<7>{});
    }
#endif
}

// The sender channels' free-slots stream ids as a runtime array (see the mux-mode note in kernel_main).
void populate_local_sender_channel_free_slots_stream_id_ordered_map(
    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids) {
    // The whole fabric table is copied, not just this router's compact count, since positions are
    // fabric-flat and a narrower router would otherwise truncate the tail.
    local_sender_channel_free_slots_stream_ids = sender_channel_free_slots_stream_ids;
}

constexpr bool IS_TEARDOWN_MASTER() { return MY_ERISC_ID == 0; }

// Ethernet keepalive: the TX queues' sequence-update timeout (ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, in cycles). A TX queue
// with nothing to send returns the acknowledgements for its paired RX queue only in a keepalive, so one-way traffic
// waits on this period once the sender's sequence window is full. The teardown master sets it on the sender and
// receiver TX queues at init and restores the firmware's values at teardown; 0 (the builder's default) leaves the
// firmware's value (8000).
constexpr uint32_t eth_txq_seq_update_timeout = get_named_compile_time_arg_val("ETH_TXQ_SEQ_UPDATE_TIMEOUT");
constexpr bool sets_seq_update_timeout = eth_txq_seq_update_timeout != 0 && IS_TEARDOWN_MASTER();
uint32_t firmware_seq_update_timeouts[2];

__attribute__((optimize("Os"))) void set_eth_txq_seq_update_timeouts() {
    if constexpr (sets_seq_update_timeout) {
        firmware_seq_update_timeouts[0] = eth_txq_reg_read(sender_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT);
        eth_txq_reg_write(sender_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, eth_txq_seq_update_timeout);
        if constexpr (receiver_txq_id != sender_txq_id) {
            firmware_seq_update_timeouts[1] = eth_txq_reg_read(receiver_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT);
            eth_txq_reg_write(receiver_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, eth_txq_seq_update_timeout);
        }
    }
}

__attribute__((optimize("Os"))) void restore_eth_txq_seq_update_timeouts() {
    if constexpr (sets_seq_update_timeout) {
        eth_txq_reg_write(sender_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, firmware_seq_update_timeouts[0]);
        if constexpr (receiver_txq_id != sender_txq_id) {
            eth_txq_reg_write(receiver_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, firmware_seq_update_timeouts[1]);
        }
    }
}

// Note: No termination check is added here intentionally. Both local ERISCs share
// the same termination signal, so both will see it and exit their respective wait
// loops naturally. Adding a termination check here would be unsafe — if one ERISC
// broke out of the sync early and skipped its scratch register write, the other
// could spin forever waiting for a value that never arrives.
__attribute__((optimize("Os"))) void wait_for_other_local_erisc() {
    constexpr uint32_t multi_erisc_sync_start_value = 0x0fed;
    constexpr uint32_t multi_erisc_sync_step2_value = 0x1bad;
    if constexpr (IS_TEARDOWN_MASTER()) {
        write_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>(multi_erisc_sync_start_value);
        while ((read_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>() & 0x1FFF) !=
               multi_erisc_sync_step2_value) {
            router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        }
        write_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>(0);
    } else {
        while ((read_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>() & 0x1FFF) !=
               multi_erisc_sync_start_value) {
            router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        }
        write_stream_scratch_register<MULTI_RISC_TEARDOWN_SYNC_STREAM_ID>(multi_erisc_sync_step2_value);
    }
}

__attribute__((optimize("Os"))) void teardown(
    volatile tt_l1_ptr tt::tt_fabric::TerminationSignal* termination_signal_ptr,
    volatile tt_l1_ptr tt::tt_fabric::EDMStatus* edm_status_ptr,
    WriteTransactionIdTracker<
        RECEIVER_NUM_BUFFERS_ARRAY[0],
        NUM_TRANSACTION_IDS,
        RX_CH_TRID_STARTS[0],
        edm_to_local_chip_noc,
        edm_to_downstream_noc> receiver_channel_0_trid_tracker
#if defined(FABRIC_2D_VC1_ACTIVE)
    ,
    WriteTransactionIdTracker<
        RECEIVER_NUM_BUFFERS_ARRAY[1],
        NUM_TRANSACTION_IDS,
        RX_CH_TRID_STARTS[1],
        edm_to_local_chip_noc,
        edm_to_downstream_noc> receiver_channel_1_trid_tracker
#endif
#if defined(FABRIC_2D_VC2_SERVICED)
    ,
    WriteTransactionIdTracker<
        RECEIVER_NUM_BUFFERS_ARRAY[VC2_RECEIVER_CHANNEL],
        NUM_TRANSACTION_IDS,
        RX_CH_TRID_STARTS[VC2_RECEIVER_CHANNEL],
        edm_to_local_chip_noc,
        edm_to_downstream_noc> receiver_channel_2_trid_tracker
#endif
) {
    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }
    if constexpr (is_receiver_channel_serviced[0]) {
        receiver_channel_0_trid_tracker.all_buffer_slot_transactions_acked();
    }
#if defined(FABRIC_2D_VC1_ACTIVE)
    if constexpr (is_receiver_channel_serviced[1]) {
        receiver_channel_1_trid_tracker.all_buffer_slot_transactions_acked();
    }
#endif
#if defined(FABRIC_2D_VC2_SERVICED)
    if constexpr (is_receiver_channel_serviced[VC2_RECEIVER_CHANNEL]) {
        receiver_channel_2_trid_tracker.all_buffer_slot_transactions_acked();
    }
#endif

    static_assert(noc_mode != DM_DYNAMIC_NOC, "Dynamic noc mode not supported");
    // re-init the noc counters as the noc api used is not incrementing them
    ncrisc_noc_counters_init();

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }
    if constexpr (wait_for_host_signal) {
        if constexpr (is_local_handshake_master) {
            notify_subordinate_routers(
                edm_channels_mask,
                local_handshake_master_eth_chan,
                (uint32_t)termination_signal_ptr,
                *termination_signal_ptr);
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();

#ifdef ARCH_BLACKHOLE
    // Tagged write completion does not reset command-buffer transaction IDs.
    // Restore this ERISC's dedicated NoC before the kernel-return checks.
    noc_clear_packet_tags(noc_index);
#endif

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }
    restore_eth_txq_seq_update_timeouts();
    if constexpr (IS_TEARDOWN_MASTER()) {
        *edm_status_ptr = tt::tt_fabric::EDMStatus::TERMINATED;
    }
}

__attribute__((optimize("Os"))) void initialize_state_for_txq1_active_mode() {
    eth_enable_packet_mode(receiver_txq_id);
    for (size_t i = 0; i < NUM_RECEIVER_CHANNELS; i++) {
        reinterpret_cast<volatile uint32_t*>(local_receiver_ack_counters_base_address)[i] = 0;
        reinterpret_cast<volatile uint32_t*>(local_receiver_completion_counters_base_address)[i] = 0;
    }
    eth_txq_reg_write(receiver_txq_id, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD);
}
__attribute__((optimize("Os"))) void initialize_state_for_txq1_active_mode_sender_side() {
    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        reinterpret_cast<volatile uint32_t*>(to_sender_remote_ack_counters_base_address)[i] = 0;
        reinterpret_cast<volatile uint32_t*>(to_sender_remote_completion_counters_base_address)[i] = 0;
    }
}

// Initialize fabric telemetry structure in L1
// This must be called early in kernel_main to ensure telemetry is properly initialized
void initialize_fabric_telemetry() {
    volatile tt_l1_ptr FabricTelemetry* fabric_telemetry =
        reinterpret_cast<volatile tt_l1_ptr FabricTelemetry*>(eth_l1_mem::address_map::AERISC_FABRIC_TELEMETRY_ADDR);
    const auto* routing_table_l1 = reinterpret_cast<tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    // Zero the entire telemetry structure to clear any garbage values
    memset(const_cast<FabricTelemetry*>(fabric_telemetry), 0, sizeof(FabricTelemetry));

    // Populate static_info fields with compile-time topology information
    fabric_telemetry->static_info.mesh_id = routing_table_l1->my_mesh_id;
    fabric_telemetry->static_info.device_id = routing_table_l1->my_device_id;
    fabric_telemetry->static_info.direction = static_cast<uint8_t>(my_direction);
    fabric_telemetry->static_info.version = FABRIC_TELEMETRY_VERSION;
    fabric_telemetry->static_info.fabric_config = 0;
    // Initialize neighbor info to sentinel values (will be populated during handshake)
    // Using 0xFFFF/0xFF as sentinel to distinguish "not yet set" from valid value 0
    fabric_telemetry->static_info.neighbor_mesh_id = 0xFFFF;
    fabric_telemetry->static_info.neighbor_device_id = 0xFF;
    // Set supported_stats bitmask to enable all telemetry features
    fabric_telemetry->static_info.supported_stats = static_cast<DynamicStatistics>(
        DynamicStatistics::BANDWIDTH | DynamicStatistics::ROUTER_STATE | DynamicStatistics::HEARTBEAT_TX |
        DynamicStatistics::HEARTBEAT_RX);

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        // In dual-ERISC mode, each ERISC initializes its own entry
        fabric_telemetry->dynamic_info.erisc[MY_ERISC_ID].router_state = RouterState::INITIALIZING;
    } else {
        // In single-ERISC mode, initialize both entries (only [0] will be used)
        fabric_telemetry->dynamic_info.erisc[0].router_state = RouterState::INITIALIZING;
        fabric_telemetry->dynamic_info.erisc[1].router_state = RouterState::INITIALIZING;
    }
}

// The same condition as receiver_erisc::runs' first test: run_kernel is not defined where it can never run (a VC2
// build's teardown takes a third tracker, and GCC checks the non-dependent call below even in an unused template).
#if defined(ARCH_BLACKHOLE) && !defined(FABRIC_2D_VC2_SERVICED)
namespace receiver_erisc {

// ERISC 1's whole kernel when `runs`: kernel_main's init steps that concern a receiver ERISC, in the same order and
// with the same ERISC syncs (six in init, three in teardown; kernel_main must keep the same count), then the main loop
// and teardown. ERISC 0 owns the sender-side state (sender channels, their worker interfaces and location info) and
// initializes it; this ERISC does not touch it. Neither ERISC zeroes the headers at the far receiver's slot addresses
// in local L1 (kernel_main's remote_receiver_channels.init does on other builds): the far router zeroes its own
// receiver headers, and on a link whose two ends share a layout these are the local receiver headers, zeroed here.
template <bool INSTANTIATED_ON_USE = true>
FORCE_INLINE void run_kernel() {
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        channel_trimming_usage_recorder.reset();
    }

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::INITIALIZATION_STARTED);
#endif

    const auto* routing_table_l1 = reinterpret_cast<tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    auto* state_manager_l1 = const_cast<tt_l1_ptr RouterStateManager*>(&routing_table_l1->state_manager);
    state_manager_l1->state = RouterState::INITIALIZING;

    set_l1_data_cache<ENABLE_RISC_CPU_DATA_CACHE>();

    // Initialize fabric telemetry early to ensure valid values before router starts
    initialize_fabric_telemetry();

    eth_txq_reg_write(sender_txq_id, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD);
    set_eth_txq_seq_update_timeouts();
    static_assert(receiver_txq_id == 1 && sender_txq_id == 0);
    // This ERISC runs the receiver TX queue (completion credits).
    initialize_state_for_txq1_active_mode();
#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::TXQ_INITIALIZED);
#endif

    // Stream registers for credit management across the Ethernet link, written by both ERISCs before the handshake
    // (as kernel_main).
    init_ptr_val<to_receiver_packets_sent_streams[0]>(0);
    init_stream_reg_if_used<to_sender_packets_acked_streams[0]>(0);
    init_stream_reg_if_used<to_sender_packets_acked_streams[1]>(0);
    init_stream_reg_if_used<to_sender_packets_completed_streams[0]>(0);
    init_stream_reg_if_used<to_sender_packets_completed_streams[1]>(0);
    init_ptr_val<sender_channel_free_slots_stream_ids[0]>(SENDER_NUM_BUFFERS_ARRAY[0]);
    init_ptr_val<sender_channel_free_slots_stream_ids[1]>(SENDER_NUM_BUFFERS_ARRAY[1]);

    wait_for_other_local_erisc();

    if constexpr (is_2d_fabric) {
        init_stream_reg_if_used<to_receiver_packets_sent_streams[1]>(0);
        init_stream_reg_if_used<to_receiver_packets_sent_streams[2]>(0);
        init_stream_reg_if_used<to_sender_packets_acked_streams[2]>(0);
        init_stream_reg_if_used<to_sender_packets_acked_streams[3]>(0);
        init_stream_reg_if_used<to_sender_packets_acked_streams[4]>(0);

        [&]<size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                 init_stream_reg_if_used<to_sender_packets_completed_streams[Is + 2]>(0);
                 if constexpr (services_fabric_position(Is + 2)) {
                     init_ptr_val<sender_channel_free_slots_stream_ids[Is + 2]>(
                         SENDER_NUM_BUFFERS_ARRAY[compact_sender_for_fabric_position(Is + 2)]);
                 }
             }()),
             ...);
        }(std::make_index_sequence<8>{});
    }

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::STREAM_REG_INITIALIZED);
#endif

    if constexpr (code_profiling_enabled_timers_bitfield != 0) {
        clear_code_profiling_buffer(code_profiling_buffer_base_addr);
    }

    volatile auto termination_signal_ptr =
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr);
    [[maybe_unused]] volatile auto edm_local_sync_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(edm_local_sync_ptr_addr);
    volatile auto edm_status_ptr = reinterpret_cast<volatile tt_l1_ptr tt::tt_fabric::EDMStatus*>(edm_status_ptr_addr);

    // Runtime args, in kernel_main's layout. The sender channels' connection semaphores and buffer-index words come
    // first; ERISC 1 does not read them.
    size_t arg_idx = 2 * MAX_NUM_SENDER_CHANNELS;
    const uint32_t has_downstream_edm_vc0_buffer_connection = get_arg_val<uint32_t>(arg_idx++);
#if defined(FABRIC_2D)
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_buffer_base_addresses;
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC0; i++) {
        downstream_edm_vc0_buffer_base_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }
#else
    std::array<uint32_t, 1> downstream_edm_vc0_buffer_base_addresses;
    downstream_edm_vc0_buffer_base_addresses[0] = get_arg_val<uint32_t>(arg_idx++);
#endif
    const uint32_t downstream_edm_vc0_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t downstream_edm_vc0_noc_y = get_arg_val<uint32_t>(arg_idx++);
#if defined(FABRIC_2D)
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_worker_registration_ids;
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_worker_location_info_addresses;
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC0; i++) {
        downstream_edm_vc0_worker_registration_ids[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC0; i++) {
        downstream_edm_vc0_worker_location_info_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    // Buffer-index semaphore addresses (unused by router-to-router connections).
    arg_idx += NUM_DOWNSTREAM_SENDERS_VC0;
#else
    std::array<uint32_t, 1> downstream_edm_vc0_worker_registration_ids;
    std::array<uint32_t, 1> downstream_edm_vc0_worker_location_info_addresses;
    downstream_edm_vc0_worker_registration_ids[0] = get_arg_val<uint32_t>(arg_idx++);
    downstream_edm_vc0_worker_location_info_addresses[0] = get_arg_val<uint32_t>(arg_idx++);
    arg_idx += 1;
#endif
#if defined(FABRIC_2D_VC1_ACTIVE)
    const uint32_t has_downstream_edm_vc1_buffer_connection = get_arg_val<uint32_t>(arg_idx++);
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC1> downstream_edm_vc1_buffer_base_addresses;
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC1; i++) {
        downstream_edm_vc1_buffer_base_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    const uint32_t downstream_edm_vc1_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t downstream_edm_vc1_noc_y = get_arg_val<uint32_t>(arg_idx++);
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC1> downstream_edm_vc1_worker_registration_ids;
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC1> downstream_edm_vc1_worker_location_info_addresses;
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC1; i++) {
        downstream_edm_vc1_worker_registration_ids[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC1; i++) {
        downstream_edm_vc1_worker_location_info_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    arg_idx += NUM_DOWNSTREAM_SENDERS_VC1;
#endif
    // Teardown semaphores, one per downstream connection (VC0's, then VC1's).
    std::array<size_t, NUM_DOWNSTREAM_CHANNELS> local_sem_for_teardown_from_downstream_edm;
    for (size_t i = 0; i < NUM_DOWNSTREAM_CHANNELS; i++) {
        local_sem_for_teardown_from_downstream_edm[i] = get_arg_val<uint32_t>(arg_idx++);
    }

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::STARTED);
#endif
    *edm_status_ptr = tt::tt_fabric::EDMStatus::STARTED;

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::DOWNSTREAM_EDM_SETUP_STARTED);
#endif
    std::array<DownstreamCursor, DownstreamConnections<0>::size> downstreams_vc0;
#if defined(FABRIC_2D_VC1_ACTIVE)
    std::array<DownstreamCursor, DownstreamConnections<1>::size> downstreams_vc1;
#endif
    const DownstreamSetup downstream_vcs[] = {
        downstream_setup<0>(
            downstreams_vc0,
            has_downstream_edm_vc0_buffer_connection,
            downstream_edm_vc0_buffer_base_addresses.data(),
            downstream_edm_vc0_worker_location_info_addresses.data(),
            downstream_edm_vc0_worker_registration_ids.data(),
            local_sem_for_teardown_from_downstream_edm.data(),
            downstream_edm_vc0_noc_x,
            downstream_edm_vc0_noc_y),
#if defined(FABRIC_2D_VC1_ACTIVE)
        downstream_setup<1>(
            downstreams_vc1,
            has_downstream_edm_vc1_buffer_connection,
            downstream_edm_vc1_buffer_base_addresses.data(),
            downstream_edm_vc1_worker_location_info_addresses.data(),
            downstream_edm_vc1_worker_registration_ids.data(),
            local_sem_for_teardown_from_downstream_edm.data() + NUM_DOWNSTREAM_SENDERS_VC0,
            downstream_edm_vc1_noc_x,
            downstream_edm_vc1_noc_y),
#endif
    };
    constexpr uint32_t num_downstream_vcs = sizeof(downstream_vcs) / sizeof(downstream_vcs[0]);
    init_downstreams(downstream_vcs, num_downstream_vcs);
#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::EDM_VCS_SETUP_COMPLETE);
#endif

    // initialize the local receiver channel buffers
    auto local_receiver_channels = tt::tt_fabric::
        ReceiverChannelBuffersFromAllocs<PACKET_HEADER_TYPE, channel_allocs, RECEIVER_TO_ENTRY_IDX>::make();
    local_receiver_channels.init<channel_allocs>(channel_buffer_size, sizeof(PACKET_HEADER_TYPE));

    WriteTransactionIdTracker<
        RECEIVER_NUM_BUFFERS_ARRAY[0],
        NUM_TRANSACTION_IDS,
        RX_CH_TRID_STARTS[0],
        edm_to_local_chip_noc,
        edm_to_downstream_noc>
        receiver_channel_0_trid_tracker;
    receiver_channel_0_trid_tracker.init();
#if defined(FABRIC_2D_VC1_ACTIVE)
    WriteTransactionIdTracker<
        RECEIVER_NUM_BUFFERS_ARRAY[1],
        NUM_TRANSACTION_IDS,
        RX_CH_TRID_STARTS[1],
        edm_to_local_chip_noc,
        edm_to_downstream_noc>
        receiver_channel_1_trid_tracker;
    receiver_channel_1_trid_tracker.init();
#endif

    wait_for_other_local_erisc();
    if constexpr (enable_ethernet_handshake) {
        if constexpr (is_handshake_sender) {
            erisc::datamover::handshake::fabric_sender_side_handshake<ENABLE_RISC_CPU_DATA_CACHE>(
                handshake_addr,
                routing_table_l1->my_mesh_id,
                routing_table_l1->my_device_id,
                termination_signal_ptr,
                DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT);
        } else {
            erisc::datamover::handshake::fabric_receiver_side_handshake<ENABLE_RISC_CPU_DATA_CACHE>(
                handshake_addr,
                routing_table_l1->my_mesh_id,
                routing_table_l1->my_device_id,
                termination_signal_ptr,
                DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT);
        }

        {
            volatile tt_l1_ptr auto* handshake_info =
                reinterpret_cast<volatile tt_l1_ptr erisc::datamover::handshake::handshake_info_t*>(handshake_addr);
            volatile tt_l1_ptr FabricTelemetry* fabric_telemetry =
                reinterpret_cast<volatile tt_l1_ptr FabricTelemetry*>(
                    eth_l1_mem::address_map::AERISC_FABRIC_TELEMETRY_ADDR);
            fabric_telemetry->static_info.neighbor_mesh_id = handshake_info->neighbor_mesh_id;
            fabric_telemetry->static_info.neighbor_device_id = handshake_info->neighbor_device_id;
        }

        *edm_status_ptr = tt::tt_fabric::EDMStatus::REMOTE_HANDSHAKE_COMPLETE;

        if constexpr (wait_for_host_signal) {
            if constexpr (is_local_handshake_master) {
                wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(
                    (uint32_t)edm_local_sync_ptr, num_local_edms - 1, termination_signal_ptr);
                constexpr uint32_t exclude_eth_chan = std::numeric_limits<uint32_t>::max();
                notify_subordinate_routers(
                    edm_channels_mask, exclude_eth_chan, (uint32_t)edm_local_sync_ptr, num_local_edms);
            } else {
                notify_master_router(local_handshake_master_eth_chan, (uint32_t)edm_local_sync_ptr);
                wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(
                    (uint32_t)edm_local_sync_ptr, num_local_edms, termination_signal_ptr);
            }

            *edm_status_ptr = tt::tt_fabric::EDMStatus::LOCAL_HANDSHAKE_COMPLETE;

            wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(
                (uint32_t)edm_status_ptr, tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC, termination_signal_ptr);

            if constexpr (is_local_handshake_master) {
                notify_subordinate_routers(
                    edm_channels_mask,
                    local_handshake_master_eth_chan,
                    (uint32_t)edm_status_ptr,
                    tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC);
            }
        }
    }

    wait_for_other_local_erisc();
#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::ETHERNET_HANDSHAKE_COMPLETE);
#endif
    // With the tensix extension, wait for the downstream tensix to be ready before opening downstream connections.
    if constexpr (num_ds_or_local_tensix_connections) {
        wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(
            (uint32_t)edm_local_tensix_sync_ptr_addr, num_ds_or_local_tensix_connections, termination_signal_ptr);
    }

    open_downstreams(downstream_vcs, num_downstream_vcs);

    wait_for_other_local_erisc();
#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::VCS_OPENED);
#endif
#if defined(FABRIC_2D)
    constexpr bool vc0_stateful_forwarding = use_2d_single_live_stateful_noc<VC0_DOWNSTREAM_EDM_SIZE>();
#else
    constexpr bool vc0_stateful_forwarding = false;
#endif
    setup_downstream_cmd_bufs(downstream_vcs, num_downstream_vcs, vc0_stateful_forwarding);

    wait_for_other_local_erisc();
#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::ROUTING_TABLE_INITIALIZED);
#endif
    // ERISC 1 has no static (router-to-router) sender connections to wait for.
    WAYPOINT("FSCW");
    WAYPOINT("FSCD");
    wait_for_other_local_erisc();
#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::INITIALIZATION_COMPLETE);
#endif

    run_main_loop(
        downstreams_vc0
#if defined(FABRIC_2D_VC1_ACTIVE)
        ,
        downstreams_vc1
#endif
    );
    WAYPOINT("LPDN");
    teardown(
        termination_signal_ptr,
        edm_status_ptr,
        receiver_channel_0_trid_tracker
#if defined(FABRIC_2D_VC1_ACTIVE)
        ,
        receiver_channel_1_trid_tracker
#endif
    );
    WAYPOINT("TEDN");
}

}  // namespace receiver_erisc
#endif

void kernel_main() {
#if defined(ARCH_BLACKHOLE) && !defined(FABRIC_2D_VC2_SERVICED)
    // ERISC 1 of a two-ERISC router may run its own kernel; everything below is every other build's.
    if constexpr (receiver_erisc::runs) {
        receiver_erisc::run_kernel();
        return;
    }
#endif
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        channel_trimming_usage_recorder.reset();
    }

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::INITIALIZATION_STARTED);
#endif

    const auto* routing_table_l1 = reinterpret_cast<tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    auto* state_manager_l1 = const_cast<tt_l1_ptr RouterStateManager*>(&routing_table_l1->state_manager);
    state_manager_l1->state = RouterState::INITIALIZING;

    set_l1_data_cache<ENABLE_RISC_CPU_DATA_CACHE>();

    // Initialize fabric telemetry early to ensure valid values before router starts
    initialize_fabric_telemetry();

    eth_txq_reg_write(sender_txq_id, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD);
    set_eth_txq_seq_update_timeouts();
    static_assert(
        receiver_txq_id == sender_txq_id || receiver_txq_id == 1,
        "For multi-txq mode, the only currently supported configuration is sender_txq_id=0 and receiver_txq_id=1");
    if constexpr (receiver_txq_id != sender_txq_id) {
        constexpr bool is_erisc_that_sets_up_second_txq = any_receiver_channel_serviced;
        if constexpr (is_erisc_that_sets_up_second_txq) {
            initialize_state_for_txq1_active_mode();
        }
        if constexpr (any_sender_channel_serviced) {
            initialize_state_for_txq1_active_mode_sender_side();
        }
    }
#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::TXQ_INITIALIZED);
#endif

    // Initialize stream register state for credit management across the Ethernet link.
    // We make sure to do this before we handshake to guarantee that the registers are
    // initialized before the other side has any possibility of modifying them.
    init_ptr_val<to_receiver_packets_sent_streams[0]>(0);
    init_stream_reg_if_used<to_sender_packets_acked_streams[0]>(0);
    init_stream_reg_if_used<to_sender_packets_acked_streams[1]>(0);
    init_stream_reg_if_used<to_sender_packets_completed_streams[0]>(0);
    init_stream_reg_if_used<to_sender_packets_completed_streams[1]>(0);
    init_ptr_val<sender_channel_free_slots_stream_ids[0]>(SENDER_NUM_BUFFERS_ARRAY[0]);
    init_ptr_val<sender_channel_free_slots_stream_ids[1]>(SENDER_NUM_BUFFERS_ARRAY[1]);

    // Init retrain sync reg
    if (IS_RETRAIN_SYNC_MASTER()) {
        write_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>(
            static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::NORMAL_EXECUTION));
    }

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }

    if constexpr (is_2d_fabric) {
        // Receiver channels 1 and 2 are only populated when a VC densifies onto them, so guard the
        // init rather than writing through an unassigned entry.
        init_stream_reg_if_used<to_receiver_packets_sent_streams[1]>(0);
        init_stream_reg_if_used<to_receiver_packets_sent_streams[2]>(0);
        init_stream_reg_if_used<to_sender_packets_acked_streams[2]>(0);
        init_stream_reg_if_used<to_sender_packets_acked_streams[3]>(0);
        init_stream_reg_if_used<to_sender_packets_acked_streams[4]>(0);

        [&]<size_t... Is>(std::index_sequence<Is...>) {
            (([&]() {
                 init_stream_reg_if_used<to_sender_packets_completed_streams[Is + 2]>(0);
                 if constexpr (services_fabric_position(Is + 2)) {
                     init_ptr_val<sender_channel_free_slots_stream_ids[Is + 2]>(
                         SENDER_NUM_BUFFERS_ARRAY[compact_sender_for_fabric_position(Is + 2)]);
                 }
             }()),
             ...);
        }(std::make_index_sequence<8>{});
    }

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::STREAM_REG_INITIALIZED);
#endif

    if constexpr (code_profiling_enabled_timers_bitfield != 0) {
        clear_code_profiling_buffer(code_profiling_buffer_base_addr);
    }

    // TODO: CONVERT TO SEMAPHORE
    volatile auto termination_signal_ptr =
        reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr);
    volatile auto edm_local_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(edm_local_sync_ptr_addr);
    volatile auto edm_status_ptr = reinterpret_cast<volatile tt_l1_ptr tt::tt_fabric::EDMStatus*>(edm_status_ptr_addr);

    size_t arg_idx = 0;
    ///////////////////////
    // Common runtime args:
    ///////////////////////
    // Read the fixed maximum sender-channel connection semaphore array.
    std::array<size_t, MAX_NUM_SENDER_CHANNELS> local_sender_channel_connection_semaphore_addrs;
    for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++) {
        local_sender_channel_connection_semaphore_addrs[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    // Read the fixed maximum sender-channel connection buffer-index array.
    std::array<size_t, MAX_NUM_SENDER_CHANNELS> local_sender_channel_connection_buffer_index_ids;
    for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++) {
        local_sender_channel_connection_buffer_index_ids[i] = get_arg_val<uint32_t>(arg_idx++);
    }

    // downstream EDM VC0 connection info
    const auto has_downstream_edm_vc0_buffer_connection = get_arg_val<uint32_t>(arg_idx++);

#if defined(FABRIC_2D)
    [[maybe_unused]] std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_buffer_base_addresses;
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC0; i++) {
        downstream_edm_vc0_buffer_base_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }
#else
    [[maybe_unused]] const auto downstream_edm_vc0_buffer_base_address = get_arg_val<uint32_t>(arg_idx++);
#endif

    [[maybe_unused]] const auto downstream_edm_vc0_noc_x = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const auto downstream_edm_vc0_noc_y = get_arg_val<uint32_t>(arg_idx++);

#if defined(FABRIC_2D)
    [[maybe_unused]] std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_worker_registration_ids;
    [[maybe_unused]] std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_worker_location_info_addresses;
    [[maybe_unused]] std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0>
        downstream_edm_vc0_buffer_index_semaphore_addresses;
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC0; i++) {
        downstream_edm_vc0_worker_registration_ids[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC0; i++) {
        downstream_edm_vc0_worker_location_info_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC0; i++) {
        downstream_edm_vc0_buffer_index_semaphore_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }
#else
    [[maybe_unused]] const auto downstream_edm_vc0_worker_registration_id = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const auto downstream_edm_vc0_worker_location_info_address = get_arg_val<uint32_t>(arg_idx++);
    [[maybe_unused]] const auto downstream_edm_vc0_buffer_index_semaphore_address = get_arg_val<uint32_t>(arg_idx++);
#endif

#if defined(FABRIC_2D_VC1_ACTIVE)
    const auto has_downstream_edm_vc1_buffer_connection = get_arg_val<uint32_t>(arg_idx++);
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC1> downstream_edm_vc1_buffer_base_addresses;
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC1; i++) {
        downstream_edm_vc1_buffer_base_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }

    const auto downstream_edm_vc1_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_edm_vc1_noc_y = get_arg_val<uint32_t>(arg_idx++);

    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC1> downstream_edm_vc1_worker_registration_ids;
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC1> downstream_edm_vc1_worker_location_info_addresses;
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC1> downstream_edm_vc1_buffer_index_semaphore_addresses;
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC1; i++) {
        downstream_edm_vc1_worker_registration_ids[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC1; i++) {
        downstream_edm_vc1_worker_location_info_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC1; i++) {
        downstream_edm_vc1_buffer_index_semaphore_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }
#endif
    [[maybe_unused]] const auto downstream_vc0_noc_interface_buffer_index_local_addr = 0;
    const auto downstream_vc1_noc_interface_buffer_index_local_addr = 0;

    // Read teardown semaphores (MAX_NUM_SENDER_CHANNELS channels)
    std::array<size_t, MAX_NUM_SENDER_CHANNELS> my_sem_for_teardown_from_edm;
    for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++) {
        my_sem_for_teardown_from_edm[i] = get_arg_val<uint32_t>(arg_idx++);
    }

    ///////////////////////////////////////////////
    // Local tensix (relay) connection runtime args
    // UDM mode only - packed at end of runtime args
    ///////////////////////////////////////////////
    const auto has_local_tensix_relay_connection = get_arg_val<uint32_t>(arg_idx++);
    uint32_t local_tensix_relay_buffer_base_address = 0;
    uint32_t local_tensix_relay_noc_x = 0;
    uint32_t local_tensix_relay_noc_y = 0;
    uint32_t local_tensix_relay_worker_registration_id = 0;
    uint32_t local_tensix_relay_worker_location_info_address = 0;
    uint32_t local_tensix_relay_free_slots_stream_id = 0;
    uint32_t local_tensix_relay_connection_buffer_index_id = 0;
    if constexpr (udm_mode) {
        if (has_local_tensix_relay_connection) {
            local_tensix_relay_buffer_base_address = get_arg_val<uint32_t>(arg_idx++);
            local_tensix_relay_noc_x = get_arg_val<uint32_t>(arg_idx++);
            local_tensix_relay_noc_y = get_arg_val<uint32_t>(arg_idx++);
            local_tensix_relay_worker_registration_id = get_arg_val<uint32_t>(arg_idx++);
            local_tensix_relay_worker_location_info_address = get_arg_val<uint32_t>(arg_idx++);
            local_tensix_relay_free_slots_stream_id = get_arg_val<uint32_t>(arg_idx++);
            local_tensix_relay_connection_buffer_index_id = get_arg_val<uint32_t>(arg_idx++);
        }
    }

    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (([&]<size_t I>() {
             if constexpr (is_sender_channel_serviced[I]) {
                 *reinterpret_cast<volatile uint32_t*>(local_sender_channel_connection_semaphore_addrs[I]) = 0;
                 // Zero the whole SenderChannelProducerCursor block, not just its first word: the
                 // producer reads the block back in one NOC read at connection open, so every field
                 // must be initialized here. This is the only time the router touches it.
                 auto* const producer_cursor =
                     reinterpret_cast<volatile uint32_t*>(local_sender_channel_connection_buffer_index_ids[I]);
                 for (size_t w = 0; w < sizeof(tt::tt_fabric::SenderChannelProducerCursor) / sizeof(uint32_t); ++w) {
                     producer_cursor[w] = 0;
                 }
             }
         }.template operator()<Is>()),
         ...);
    }(std::make_index_sequence<NUM_SENDER_CHANNELS>{});

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::STARTED);
#endif
    *edm_status_ptr = tt::tt_fabric::EDMStatus::STARTED;

    // Hack for mux mode until all remaining VC1 logic is removed from fabric
    // Needed so `downstream_edm_noc_interfaces_vc0` can be initialized properly below
    // Issue #33360 TODO: Create a new array for downstream receiver stream IDs
    // so we can remove this hack.
    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> local_sender_channel_free_slots_stream_ids;

    [[maybe_unused]] const auto& local_sem_for_teardown_from_downstream_edm =
        take_first_n_elements<NUM_DOWNSTREAM_CHANNELS, MAX_NUM_SENDER_CHANNELS, size_t>(my_sem_for_teardown_from_edm);

    auto remote_receiver_channels = tt::tt_fabric::ReceiverChannelBuffersFromAllocs<
        PACKET_HEADER_TYPE,
        eth_remote_channel_allocs,
        REMOTE_RECEIVER_TO_ENTRY_IDX>::make();

    auto local_receiver_channels = tt::tt_fabric::
        ReceiverChannelBuffersFromAllocs<PACKET_HEADER_TYPE, channel_allocs, RECEIVER_TO_ENTRY_IDX>::make();

    auto local_sender_channels =
        tt::tt_fabric::SenderChannelBuffersFromAllocs<PACKET_HEADER_TYPE, channel_allocs, SENDER_TO_ENTRY_IDX>::make();

    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_live_semaphore_addresses =
        take_first_n_elements<NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, size_t>(
            local_sender_channel_connection_semaphore_addrs);
    std::array<size_t, NUM_SENDER_CHANNELS> local_sender_connection_info_addresses =
        take_first_n_elements<NUM_SENDER_CHANNELS, MAX_NUM_SENDER_CHANNELS, size_t>(
            std::array<size_t, MAX_NUM_SENDER_CHANNELS>{
                local_sender_channel_0_connection_info_addr,
                local_sender_channel_1_connection_info_addr,
                local_sender_channel_2_connection_info_addr,
                local_sender_channel_3_connection_info_addr,
                local_sender_channel_4_connection_info_addr,
                local_sender_channel_5_connection_info_addr,
                local_sender_channel_6_connection_info_addr,
                local_sender_channel_7_connection_info_addr,
                local_sender_channel_8_connection_info_addr,
                local_sender_channel_9_connection_info_addr});

    for (size_t i = 0; i < NUM_SENDER_CHANNELS; i++) {
        auto connection_worker_info_ptr = reinterpret_cast<volatile tt::tt_fabric::EDMChannelWorkerLocationInfo*>(
            local_sender_connection_info_addresses[i]);
        connection_worker_info_ptr->edm_read_counter = 0;
    }
    // create the sender channel worker interfaces with input array of number of buffers
    auto local_sender_channel_worker_interfaces =
        tt::tt_fabric::EdmChannelWorkerInterfaces<tt::tt_fabric::worker_handshake_noc, SENDER_NUM_BUFFERS_ARRAY>::make(
            std::make_index_sequence<NUM_SENDER_CHANNELS>{});

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::DOWNSTREAM_EDM_SETUP_STARTED);
#endif

    // TODO: change to TMP.
    std::array<RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC0>, VC0_DOWNSTREAM_EDM_SIZE>
        downstream_edm_noc_interfaces_vc0;

    // Ensure array size is at least 1 to avoid undefined behavior
    static_assert(VC0_DOWNSTREAM_EDM_SIZE > 0, "VC0_DOWNSTREAM_EDM_SIZE must be at least 1");
#if defined(FABRIC_2D)
    [[maybe_unused]] constexpr bool use_vc0_single_live_stateful_noc =
        use_2d_single_live_stateful_noc<VC0_DOWNSTREAM_EDM_SIZE>();
#else
    [[maybe_unused]] constexpr bool use_vc0_single_live_stateful_noc = false;
#endif

    populate_local_sender_channel_free_slots_stream_id_ordered_map(local_sender_channel_free_slots_stream_ids);

    // Receiver-side setup (downstream router adapters, receiver channel buffers) belongs to the ERISC that services the
    // receiver channels. ERISC 0 running sender_erisc skips it: ERISC 1 has the same compile-time and runtime args and
    // does these writes in the same init phase, before the next ERISC sync (here, or in receiver_erisc::run_kernel).
    constexpr bool sets_up_receiver_side = !sender_erisc::runs;
    if (sets_up_receiver_side && has_downstream_edm_vc0_buffer_connection) {
        uint32_t has_downstream_edm = has_downstream_edm_vc0_buffer_connection & 0xF;
        uint32_t compact_index = 0;
        uint32_t dense_index = 0;
        while (has_downstream_edm) {
            if (has_downstream_edm & 0x1) {
                const auto teardown_sem_address = local_sem_for_teardown_from_downstream_edm[dense_index];
                *reinterpret_cast<volatile uint32_t* const>(teardown_sem_address) = 0;
#if defined(FABRIC_2D)
                auto receiver_channel_free_slots_stream_id = StreamId{vc0_local_free_slots_stream_id(compact_index)};
#else
                auto receiver_channel_free_slots_stream_id = StreamId{vc0_local_free_slots_stream_id(0)};
#endif
                new (&downstream_edm_noc_interfaces_vc0[compact_index])
                    RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC0>(
                        is_persistent_fabric,
                        (downstream_edm_vc0_noc_x >> (dense_index * 8)) & 0xFF,
                        (downstream_edm_vc0_noc_y >> (dense_index * 8)) & 0xFF,
#ifdef FABRIC_2D
                        downstream_edm_vc0_buffer_base_addresses[dense_index],
#else
                        downstream_edm_vc0_buffer_base_address,
#endif
                        DOWNSTREAM_SENDER_NUM_BUFFERS_VC0,
#ifdef FABRIC_2D
                        // connection handshake address on downstream edm
                        downstream_edm_vc0_worker_registration_ids[dense_index],
                        // worker location info address on downstream edm
                        // written by this interface when it connects to the downstream edm
                        // so that the downstream edm knows who its upstream peer is
                        downstream_edm_vc0_worker_location_info_addresses[dense_index],
#else
                        downstream_edm_vc0_worker_registration_id,
                        downstream_edm_vc0_worker_location_info_address,
#endif
                        channel_buffer_size,
#ifdef FABRIC_2D
                        downstream_edm_vc0_buffer_index_semaphore_addresses[dense_index],
#else
                        downstream_edm_vc0_buffer_index_semaphore_address,
#endif
                        0,  // unused for router-to-router connections (worker-side free-slots pointer)
                        reinterpret_cast<volatile uint32_t* const>(teardown_sem_address),
                        downstream_vc0_noc_interface_buffer_index_local_addr,
#if defined(FABRIC_2D)
#if defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)
                        get_vc1_downstream_sender_channel_free_slots_stream_id(compact_index),
#else
                        get_vc0_downstream_sender_channel_free_slots_stream_id(compact_index),
#endif
#else
                        sender_channel_1_free_slots_stream_id,
#endif
                        // This is our local stream register for the copy of the downstream router's
                        // free slots
                        receiver_channel_free_slots_stream_id,
                        receiver_channel_forwarding_data_cmd_buf_ids[0],
                        receiver_channel_forwarding_sync_cmd_buf_ids[0]);
                if constexpr (
                    NUM_ACTIVE_ERISCS == 1 && !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !use_vc0_single_live_stateful_noc) {
                    downstream_edm_noc_interfaces_vc0[compact_index]
                        .template setup_edm_noc_cmd_buf<
                            tt::tt_fabric::edm_to_downstream_noc,
                            tt::tt_fabric::forward_and_local_write_noc_vc>();
                }
                dense_index++;
            }
            compact_index++;
            has_downstream_edm >>= 1;
        }
#if defined(FABRIC_2D)
        if constexpr (NUM_ACTIVE_ERISCS == 1 && use_vc0_single_live_stateful_noc) {
            if (has_downstream_edm_vc0_buffer_connection & 0x1) {
                downstream_edm_noc_interfaces_vc0[0]
                    .template setup_edm_noc_cmd_buf<
                        tt::tt_fabric::edm_to_downstream_noc,
                        tt::tt_fabric::forward_and_local_write_noc_vc>();
            }
        }
#endif
    }

    std::array<RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC1>, VC1_DOWNSTREAM_EDM_SIZE>
        downstream_edm_noc_interfaces_vc1;

#if defined(FABRIC_2D_VC1_ACTIVE)
    // Ensure array size is at least 1 to avoid undefined behavior
    static_assert(VC1_DOWNSTREAM_EDM_SIZE > 0, "VC1_DOWNSTREAM_EDM_SIZE must be at least 1");
    if (sets_up_receiver_side && has_downstream_edm_vc1_buffer_connection) {
        uint32_t has_downstream_edm = has_downstream_edm_vc1_buffer_connection & 0xF;
        uint32_t compact_index = 0;
        uint32_t dense_index = 0;
        while (has_downstream_edm) {
            if (has_downstream_edm & 0x1) {
                const auto teardown_sem_address =
                    local_sem_for_teardown_from_downstream_edm[NUM_DOWNSTREAM_SENDERS_VC0 + dense_index];
                *reinterpret_cast<volatile uint32_t* const>(teardown_sem_address) = 0;
                new (&downstream_edm_noc_interfaces_vc1[compact_index])
                    RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC1>(
                        is_persistent_fabric,
                        (downstream_edm_vc1_noc_x >> (dense_index * 8)) & 0xFF,
                        (downstream_edm_vc1_noc_y >> (dense_index * 8)) & 0xFF,
                        downstream_edm_vc1_buffer_base_addresses[dense_index],
                        DOWNSTREAM_SENDER_NUM_BUFFERS_VC1,
                        downstream_edm_vc1_worker_registration_ids[dense_index],
                        downstream_edm_vc1_worker_location_info_addresses[dense_index],
                        channel_buffer_size,
                        downstream_edm_vc1_buffer_index_semaphore_addresses[dense_index],
                        0,  // unused for router-to-router connections (worker-side free-slots pointer)
                        reinterpret_cast<volatile uint32_t* const>(teardown_sem_address),
                        downstream_vc1_noc_interface_buffer_index_local_addr,
                        get_vc1_downstream_sender_channel_free_slots_stream_id(compact_index),
                        StreamId{vc1_local_free_slots_stream_id(compact_index)},
                        receiver_channel_forwarding_data_cmd_buf_ids[1],
                        receiver_channel_forwarding_sync_cmd_buf_ids[1]);
                dense_index++;
            }
            compact_index++;
            has_downstream_edm >>= 1;
        }
    }
#endif

    // Setup local tensix relay connection (UDM mode only)
    // This is a separate connection path from downstream EDM connections
    // Relay handles forwarding packets to local chip workers
    // Uses dedicated stream IDs and L1 locations to avoid assumptions about direction indexing
    // LOCAL_RELAY_NUM_BUFFERS comes from compile-time args (propagated from relay config)
    RouterToRouterSender<LOCAL_RELAY_NUM_BUFFERS> local_relay_interface;
    if constexpr (udm_mode) {
        if (has_local_tensix_relay_connection) {
            new (&local_relay_interface) RouterToRouterSender<LOCAL_RELAY_NUM_BUFFERS>(
                true,
                local_tensix_relay_noc_x,
                local_tensix_relay_noc_y,
                local_tensix_relay_buffer_base_address,
                LOCAL_RELAY_NUM_BUFFERS,
                local_tensix_relay_worker_registration_id,
                local_tensix_relay_worker_location_info_address,
                channel_buffer_size,
                local_tensix_relay_connection_buffer_index_id,
                0,
                nullptr,
                0,
                // Remote stream: relay's free slots stream (what relay publishes) - from runtime args
                StreamId{local_tensix_relay_free_slots_stream_id},
                // Local stream: our copy of relay's free slots - dedicated stream ID for relay
                StreamId{tensix_relay_local_free_slots_stream_id},
                receiver_channel_forwarding_data_cmd_buf_ids[0],
                receiver_channel_forwarding_sync_cmd_buf_ids[0]);

            // Only receiver channel servicing cores should be setting up the noc cmd buf.
            // Setup NOC command buffer for relay interface
            if constexpr (NUM_ACTIVE_ERISCS == 1 && !FORCE_ALL_PATHS_TO_USE_SAME_NOC) {
                local_relay_interface.template setup_edm_noc_cmd_buf<
                    tt::tt_fabric::edm_to_downstream_noc,
                    tt::tt_fabric::forward_and_local_write_noc_vc>();
            }
        }
    }

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::EDM_VCS_SETUP_COMPLETE);
#endif

    if constexpr (sets_up_receiver_side) {
        // initialize the local receiver channel buffers
        local_receiver_channels.init<channel_allocs>(channel_buffer_size, sizeof(PACKET_HEADER_TYPE));
        // The far receiver's slot addresses, for the generic loop. This also zeroes the headers at those addresses in
        // local L1, which on a link whose two ends share a layout are the local receiver headers zeroed just above.
        remote_receiver_channels.init<eth_remote_channel_allocs>(channel_buffer_size, sizeof(PACKET_HEADER_TYPE));
    }
    // initialize the local sender channel buffers
    local_sender_channels.init<channel_allocs>(channel_buffer_size, sizeof(PACKET_HEADER_TYPE));

    if constexpr (any_sender_channel_serviced) {
        init_local_sender_channel_worker_interfaces(
            local_sender_connection_live_semaphore_addresses,
            local_sender_connection_info_addresses,
            local_sender_channel_worker_interfaces);
    }

    WriteTransactionIdTracker<
        RECEIVER_NUM_BUFFERS_ARRAY[0],
        NUM_TRANSACTION_IDS,
        RX_CH_TRID_STARTS[0],
        edm_to_local_chip_noc,
        edm_to_downstream_noc>
        receiver_channel_0_trid_tracker;
    receiver_channel_0_trid_tracker.init();

#if defined(FABRIC_2D_VC1_ACTIVE)
    WriteTransactionIdTracker<
        RECEIVER_NUM_BUFFERS_ARRAY[1],
        NUM_TRANSACTION_IDS,
        RX_CH_TRID_STARTS[1],
        edm_to_local_chip_noc,
        edm_to_downstream_noc>
        receiver_channel_1_trid_tracker;
    receiver_channel_1_trid_tracker.init();
#endif

#if defined(FABRIC_2D_VC2_SERVICED)
    WriteTransactionIdTracker<
        RECEIVER_NUM_BUFFERS_ARRAY[VC2_RECEIVER_CHANNEL],
        NUM_TRANSACTION_IDS,
        RX_CH_TRID_STARTS[VC2_RECEIVER_CHANNEL],
        edm_to_local_chip_noc,
        edm_to_downstream_noc>
        receiver_channel_2_trid_tracker;
    receiver_channel_2_trid_tracker.init();
#endif

#ifdef ARCH_BLACKHOLE
    [[maybe_unused]] constexpr bool use_posted_writes_for_connection_open = false;
#else
    [[maybe_unused]] constexpr bool use_posted_writes_for_connection_open = true;
#endif

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }
    if constexpr (enable_ethernet_handshake) {
        if constexpr (is_handshake_sender) {
            erisc::datamover::handshake::fabric_sender_side_handshake<ENABLE_RISC_CPU_DATA_CACHE>(
                handshake_addr,
                routing_table_l1->my_mesh_id,
                routing_table_l1->my_device_id,
                termination_signal_ptr,
                DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT);
        } else {
            erisc::datamover::handshake::fabric_receiver_side_handshake<ENABLE_RISC_CPU_DATA_CACHE>(
                handshake_addr,
                routing_table_l1->my_mesh_id,
                routing_table_l1->my_device_id,
                termination_signal_ptr,
                DEFAULT_HANDSHAKE_CONTEXT_SWITCH_TIMEOUT);
        }

        {
            volatile tt_l1_ptr auto* handshake_info =
                reinterpret_cast<volatile tt_l1_ptr erisc::datamover::handshake::handshake_info_t*>(handshake_addr);
            volatile tt_l1_ptr FabricTelemetry* fabric_telemetry =
                reinterpret_cast<volatile tt_l1_ptr FabricTelemetry*>(
                    eth_l1_mem::address_map::AERISC_FABRIC_TELEMETRY_ADDR);
            fabric_telemetry->static_info.neighbor_mesh_id = handshake_info->neighbor_mesh_id;
            fabric_telemetry->static_info.neighbor_device_id = handshake_info->neighbor_device_id;
        }

        *edm_status_ptr = tt::tt_fabric::EDMStatus::REMOTE_HANDSHAKE_COMPLETE;

        if constexpr (wait_for_host_signal) {
            if constexpr (is_local_handshake_master) {
                wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(
                    (uint32_t)edm_local_sync_ptr, num_local_edms - 1, termination_signal_ptr);
                // This master sends notification to self for multi risc in single eth core case,
                // This still send to self even though with single risc core case, but no side effects
                constexpr uint32_t exclude_eth_chan = std::numeric_limits<uint32_t>::max();
                notify_subordinate_routers(
                    edm_channels_mask, exclude_eth_chan, (uint32_t)edm_local_sync_ptr, num_local_edms);
            } else {
                notify_master_router(local_handshake_master_eth_chan, (uint32_t)edm_local_sync_ptr);
                wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(
                    (uint32_t)edm_local_sync_ptr, num_local_edms, termination_signal_ptr);
            }

            *edm_status_ptr = tt::tt_fabric::EDMStatus::LOCAL_HANDSHAKE_COMPLETE;

            wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(
                (uint32_t)edm_status_ptr, tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC, termination_signal_ptr);

            if constexpr (is_local_handshake_master) {
                notify_subordinate_routers(
                    edm_channels_mask,
                    local_handshake_master_eth_chan,
                    (uint32_t)edm_status_ptr,
                    tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC);
            }
        }
    }

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }
#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::ETHERNET_HANDSHAKE_COMPLETE);
#endif
    // if enable the tensix extension, then before open downstream connection, need to wait for downstream tensix ready
    // for connection.
    if constexpr (num_ds_or_local_tensix_connections) {
        wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(
            (uint32_t)edm_local_tensix_sync_ptr_addr, num_ds_or_local_tensix_connections, termination_signal_ptr);
    }

    if constexpr (is_2d_fabric) {
        // Helper function to open downstream EDM connections, works for both VC0 and VC1
        auto open_downstream_edm_connections =
            [](auto& downstream_edm_noc_interfaces, uint32_t has_downstream_edm, int receiver_channel_idx) {
                uint32_t edm_index = 0;
                if (is_receiver_channel_serviced[receiver_channel_idx]) {
                    while (has_downstream_edm) {
                        if (has_downstream_edm & 0x1) {
                            // open connections with available downstream edms
                            downstream_edm_noc_interfaces[edm_index]
                                .template open<
                                    false,
                                    use_posted_writes_for_connection_open,
                                    tt::tt_fabric::worker_handshake_noc>();
                        }
                        edm_index++;
                        has_downstream_edm >>= 1;
                    }
                }
            };

        open_downstream_edm_connections(
            downstream_edm_noc_interfaces_vc0, has_downstream_edm_vc0_buffer_connection & 0xF, 0);
#if defined(FABRIC_2D_VC1_ACTIVE)
        open_downstream_edm_connections(
            downstream_edm_noc_interfaces_vc1, has_downstream_edm_vc1_buffer_connection & 0xF, 1);
#endif
        if constexpr (udm_mode) {
            if (has_local_tensix_relay_connection) {
                // open connection here to relay kernel
                local_relay_interface
                    .template open<false, use_posted_writes_for_connection_open, tt::tt_fabric::worker_handshake_noc>();
            }
        }
    } else {
        if constexpr (is_receiver_channel_serviced[0]) {
            if (has_downstream_edm_vc0_buffer_connection) {
                downstream_edm_noc_interfaces_vc0[0]
                    .template open<false, use_posted_writes_for_connection_open, tt::tt_fabric::worker_handshake_noc>();
                ASSERT(
                    get_ptr_val(downstream_edm_noc_interfaces_vc0[0].get_worker_credits_stream_id()) ==
                    DOWNSTREAM_SENDER_NUM_BUFFERS_VC0);
            }
        }
    }

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }
#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::VCS_OPENED);
#endif
    if constexpr (is_receiver_channel_serviced[0] and NUM_ACTIVE_ERISCS > 1) {
        if constexpr (!FORCE_ALL_PATHS_TO_USE_SAME_NOC) {
            uint32_t has_downstream_edm = has_downstream_edm_vc0_buffer_connection & 0xF;
            uint32_t edm_index = 0;
            while (has_downstream_edm) {
                if (has_downstream_edm & 0x1) {
                    downstream_edm_noc_interfaces_vc0[edm_index]
                        .template setup_edm_noc_cmd_buf<
                            tt::tt_fabric::edm_to_downstream_noc,
                            tt::tt_fabric::forward_and_local_write_noc_vc>();
                }
                edm_index++;
                has_downstream_edm >>= 1;
            }
#if defined(FABRIC_2D)
            if constexpr (use_vc0_single_live_stateful_noc) {
                if (has_downstream_edm_vc0_buffer_connection & 0x1) {
                    downstream_edm_noc_interfaces_vc0[0]
                        .template setup_edm_noc_cmd_buf<
                            tt::tt_fabric::edm_to_downstream_noc,
                            tt::tt_fabric::forward_and_local_write_noc_vc>();
                }
            }
#endif
        }
    }
#if defined(FABRIC_2D_VC1_ACTIVE)
    if constexpr (is_receiver_channel_serviced[1] and NUM_ACTIVE_ERISCS > 1) {
        if constexpr (!FORCE_ALL_PATHS_TO_USE_SAME_NOC) {
            uint32_t has_downstream_edm = has_downstream_edm_vc1_buffer_connection & 0xF;
            uint32_t edm_index = 0;
            while (has_downstream_edm) {
                if (has_downstream_edm & 0x1) {
                    downstream_edm_noc_interfaces_vc1[edm_index]
                        .template setup_edm_noc_cmd_buf<
                            tt::tt_fabric::edm_to_downstream_noc,
                            tt::tt_fabric::forward_and_local_write_noc_vc>();
                }
                edm_index++;
                has_downstream_edm >>= 1;
            }
        }
    }
#endif
    std::array<uint8_t, num_eth_ports> port_direction_table;

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::ROUTING_TABLE_INITIALIZED);
#endif

    WAYPOINT("FSCW");
    if constexpr (sender_erisc::runs) {
        sender_erisc::wait_for_static_connections(local_sender_channel_worker_interfaces, termination_signal_ptr);
    } else {
        wait_for_static_connection_to_ready(
            local_sender_channel_worker_interfaces, local_sender_channel_free_slots_stream_ids, termination_signal_ptr);
    }
    WAYPOINT("FSCD");

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::INITIALIZATION_COMPLETE);
#endif
    if constexpr (MY_ERISC_ID == 0) {
        state_manager_l1->state = RouterState::INITIALIZING;
    }

    //////////////////////////////
    //////////////////////////////
    //        MAIN LOOP
    //////////////////////////////
    //////////////////////////////
    // ERISC 0 of a two-ERISC router may run sender_erisc; every other build runs the generic loop.
    if constexpr (sender_erisc::runs) {
        sender_erisc::run_main_loop(local_sender_channel_worker_interfaces, local_sender_channel_free_slots_stream_ids);
    } else {
        run_fabric_edm_main_loop<
            NUM_RECEIVER_CHANNELS,
            RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC0>,
            RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC1>,
            RouterToRouterSender<LOCAL_RELAY_NUM_BUFFERS>>(
            local_receiver_channels,
            local_sender_channels,
            local_sender_channel_worker_interfaces,
            downstream_edm_noc_interfaces_vc0,
            downstream_edm_noc_interfaces_vc1,
            local_relay_interface,
            remote_receiver_channels,
            termination_signal_ptr,
            receiver_channel_0_trid_tracker,
#if defined(FABRIC_2D_VC1_ACTIVE)
            receiver_channel_1_trid_tracker,
#endif
#if defined(FABRIC_2D_VC2_SERVICED)
            receiver_channel_2_trid_tracker,
#endif
            port_direction_table,
            local_sender_channel_free_slots_stream_ids);
    }
    WAYPOINT("LPDN");
    teardown(
        termination_signal_ptr,
        edm_status_ptr,
        receiver_channel_0_trid_tracker
#if defined(FABRIC_2D_VC1_ACTIVE)
        ,
        receiver_channel_1_trid_tracker
#endif
#if defined(FABRIC_2D_VC2_SERVICED)
        ,
        receiver_channel_2_trid_tracker
#endif
    );
    WAYPOINT("TEDN");
}
