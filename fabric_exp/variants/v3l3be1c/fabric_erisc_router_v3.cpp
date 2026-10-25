// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric ERISC router, version 3.
//
// Shared fabric headers select this router's policies from FABRIC_ERISC_ROUTER_V3 (receiver completion batching in
// fabric_erisc_router_ct_args.hpp, deferred flushed atomics and the strict-flush check in
// fabric_edm_packet_transmission.hpp), so it must be defined before the first of them is included.
#define FABRIC_ERISC_ROUTER_V3 1

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "internal/ethernet/tunneling.h"

#include "fabric/fabric_edm_packet_header.hpp"
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

FORCE_INLINE void record_sender_channel_usage(size_t sender_channel_index, size_t packet_size_bytes) {
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        channel_trimming_usage_recorder.set_sender_channel_used(sender_channel_index);
        channel_trimming_usage_recorder.update_sender_channel_packet_size(
            sender_channel_index, static_cast<uint16_t>(packet_size_bytes));
    }
}

// Flushed atomics on E1 may be deferred past later packets instead of spinning at the atomic for its flush. Needs the
// batched-completion receiver (one transaction ID per slot, 16 slots or more) alone on its ERISC and one NoC for local
// and downstream writes, so that one global check covers every outstanding write.
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

FORCE_INLINE void generic_rx_local_delivery_impl(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
    uint16_t payload_size_bytes,
    tt::tt_fabric::NocSendType noc_send_type,
    uint8_t transaction_id,
    uint8_t rx_channel_id,
    uint32_t* pending_atomic_flush = nullptr) {
    execute_chip_unicast_to_local_chip_impl(
        packet_start, payload_size_bytes, noc_send_type, transaction_id, rx_channel_id, pending_atomic_flush);
}

FORCE_INLINE void generic_rx_local_delivery(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
    [[maybe_unused]] uint16_t payload_size_bytes,
    uint8_t transaction_id,
    uint8_t rx_channel_id) {
    auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_start);
    generic_rx_local_delivery_impl(
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
                    generic_rx_local_delivery_impl(
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
                    generic_rx_local_delivery_impl(
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
        generic_rx_local_delivery_impl(
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
FORCE_INLINE __attribute__((optimize("no-jump-tables"))) bool admit_2d_dispatch(
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
FORCE_INLINE __attribute__((optimize("no-jump-tables"))) uint8_t forward_2d_dispatch(
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
    EdmChannelWorkerIFs& local_sender_channel_worker_interface, uint32_t stream_id) {
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
// E0 reads the free-slot stream registers of up to four VC0 sender channels back to back at the top of each pass, so
// their latencies overlap, and each channel visit uses its value.
constexpr size_t grouped_packet_check_channels = ACTUAL_VC0_SENDER_CHANNELS < 4 ? ACTUAL_VC0_SENDER_CHANNELS : 4;

template <size_t CH>
constexpr uint32_t free_slots_reg_addr() {
    return STREAM_REG_ADDR(
        sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(CH)],
        STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
}

template <size_t CH>
FORCE_INLINE uint32_t read_free_slots_raw() {
    return *reinterpret_cast<volatile uint32_t*>(free_slots_reg_addr<CH>());
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
    // Compiler scheduling barrier: keeps all N stream-register loads back-to-back before ch0 masks raw[0].
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

// With equal power-of-two slot counts the all-free test is one bit test: the free count never exceeds the slot count.
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

// Mask for the free-slot count in a grouped value. 0x7FF fits an immediate; every channel's slot count fits under it.
constexpr uint32_t grouped_free_slots_imm_mask = 0x7FFu;
static_assert(
    grouped_free_slots_imm_mask <= ((1u << REMOTE_DEST_WORDS_FREE_WIDTH) - 1),
    "grouped_free_slots_imm_mask must fit within REMOTE_DEST_WORDS_FREE_WIDTH");

template <size_t CH>
FORCE_INLINE bool channel_all_slots_free(uint32_t raw) {
    static_assert(SENDER_NUM_BUFFERS_ARRAY[CH] <= grouped_free_slots_imm_mask);
    if constexpr (skip_idle_bit_test) {
        return (raw & SENDER_NUM_BUFFERS_ARRAY[CH]) != 0;
    } else {
        return (raw & grouped_free_slots_imm_mask) == SENDER_NUM_BUFFERS_ARRAY[CH];
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

        if (static_cast<int32_t>(pre_txq_status << (31 - ETH_TXQ_STATUS_CMD_ONGOING_BIT)) < 0) {
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

        if (static_cast<int32_t>(txq_status_raw << (31 - ETH_TXQ_STATUS_CMD_ONGOING_BIT)) < 0) {
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
    static_assert(WorkerInterfaceT::num_buffers <= grouped_free_slots_imm_mask);
    uint32_t free_slots;
    if constexpr (sender_channel_index < grouped_packet_check_channels) {
        free_slots = grouped_free_slots_raw & grouped_free_slots_imm_mask;
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
        static_assert(WorkerInterfaceT::num_buffers <= grouped_free_slots_imm_mask);
        uint32_t free_slots;
        if constexpr (sender_channel_index < grouped_packet_check_channels) {
            free_slots = grouped_free_slots_raw & grouped_free_slots_imm_mask;
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
    typename RemoteEthReceiverChannels,
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
        [[maybe_unused]] RemoteEthReceiverChannels& remote_receiver_channels,
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
                    // A deferred flushed atomic is pending. 1: strict (waits for every earlier write on the NoC).
                    // Relaxed: 0x10 | the trid of its own packet, then 0x100 | (trid_b << 4) | trid_a once the one
                    // or two trids it still waits for are known.
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
                                ((trid_s - trid) & 0xFu) > 1u;
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

// The router's main loop. Each outer iteration checks for termination (and, on the context-switch master, pending
// router commands), runs iterations_between_ctx_switch_and_teardown_checks passes over the channels, then updates
// telemetry, the heartbeat and the idle context switch. In two-ERISC mode ERISC 0 services the sender channels and
// ERISC 1 the receiver channels: each channel step compiles to nothing on the ERISC that does not service it.
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

    // Keep hot routing coordinates in address-untaken scalars so they remain in registers across the
    // main loop; the RISC data cache is disabled for this build.
    const uint8_t my_mesh_coord_y = routing_table.my_mesh_coord_y;
    const uint8_t my_mesh_coord_x = routing_table.my_mesh_coord_x;
    // Copy the mesh id as well: routing_table is passed by reference and invalidated during receiver
    // steps, while this scalar can remain in a register for the intermesh-exit comparison.
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

    // Workaround the perf regression in RingAsLinear test.
    auto outbound_to_receiver_channel_pointer_ch0 =
        outbound_to_receiver_channel_pointers.template get<VC0_RECEIVER_CHANNEL>();

    // Workaround the perf regression in RingAsLinear test.
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
#if defined(ARCH_BLACKHOLE)
    constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_ADDR = 0x7CC70;
#else
    constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_ADDR = 0x1F80;
#endif
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
                        remote_receiver_channels,
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
                        remote_receiver_channels,
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
                            remote_receiver_channels,
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
                            remote_receiver_channels,
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
                            remote_receiver_channels,
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
                        remote_receiver_channels,
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
                        remote_receiver_channels,
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
                        remote_receiver_channels,
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
                            remote_receiver_channels,
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

// ============================================================================
// E1 receive program (two-ERISC mode, the receiver ERISC)
// ============================================================================
//
// In two-ERISC mode ERISC 1 services only the receiver channels. It runs its own program, e1_kernel_main (before
// kernel_main): its part of kernel_main's init, its own downstream connections, the receive loop below, teardown. Every
// other build (ERISC 0, single ERISC, the speedy path, first-level ack or deadlock avoidance, VC2, UDM, tensix mux,
// Wormhole) runs kernel_main and run_fabric_edm_main_loop above, unchanged.
//
// Per pass, per serviced receiver channel (VC0, then VC1):
//   1. Packets known present: the packets-received stream register is read only when the count from its last read is
//      used up. E1 still decrements the register once per retired packet, so it stays the source of truth; the count
//      E1 keeps is a lower bound of it and a packet is never seen before it arrived.
//   2. Header: payload size, send type and src_ch_id in one word, then the route byte(s). The slot address is a running
//      pointer over the channel's compile-time ring, not a table.
//   3. Admission: local-only packets need nothing. A forward tests a count of known-free slots per live downstream and
//      re-reads that downstream's local free-slots register only when the count is zero. Each forward still decrements
//      the register, so it stays the source of truth and the count is a lower bound of it.
//   4. Deferred flushed atomic (VC0, 2D): drained or bypassed exactly as before, before any NoC command of this packet.
//   5. Forward: one E1-owned cursor per downstream (running slot address, compile-time command buffers and local
//      stream ids). Single-live E/W and 1D forwards are stateful, as before.
//   6. Local delivery: a plain write is issued directly (the write arm's NoC command); every other send type goes
//      through execute_chip_unicast_to_local_chip_impl, unchanged.
//   7. Retire: the slot's record (src_ch_id and the command buffers that read the slot), packets-received register -1,
//      next slot.
//   8. Completions: one credit frame per batch, as before. The batch's records say which slots' writes must have left:
//      the newest slot that reads its data, plus the newest older one on the other command buffer if any (exact for
//      batches that mix local writes and forwards, or include atomics). A full batch from one sender is one credit of 4.
//
// Receiver channel 0 always uses the inline compile-time forms below, also on VC1 routers; only the VC1 receiver channel
// goes through the shared out-of-line bodies (VC1 code size).
//
// Grouped reads (several register reads issued back to back so their latencies overlap) use v3a's form: volatile
// loads between two std::atomic_signal_fence(seq_cst), and their results combined only after the second fence.

constexpr bool e1_receive_program = [] {
#if defined(ARCH_BLACKHOLE) && !defined(FABRIC_2D_VC2_SERVICED)
    if (!(NUM_ACTIVE_ERISCS == 2 && MY_ERISC_ID == 1)) {
        return false;
    }
    bool ok = !any_sender_channel_serviced && is_receiver_channel_serviced[0] && !super_speedy_mode &&
              multi_txq_enabled && !enable_deadlock_avoidance && !ENABLE_FIRST_LEVEL_ACK_VC0 &&
              !ENABLE_FIRST_LEVEL_ACK_VC1 && !udm_mode && !skip_src_ch_id_update && !enable_context_switch &&
              !enable_trid_flush_check_on_noc_txn && !FORCE_ALL_PATHS_TO_USE_SAME_NOC &&
              tt::tt_fabric::local_chip_noc_equals_downstream_noc && tt::tt_fabric::edm_to_local_chip_noc == noc_index &&
              (is_2d_fabric || std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::LowLatencyRoutingFields>);
    for (size_t ch = 0; ch < NUM_RECEIVER_CHANNELS; ch++) {
        if (is_receiver_channel_serviced[ch]) {
            ok = ok && receiver_channel_uses_counter_credits(ch) && !is_receiver_channel_forwarding_disabled[ch];
        }
    }
    return ok;
#else
    return false;
#endif
}();

#if defined(ARCH_BLACKHOLE)
// Packet and free-slot counts in stream registers never exceed a channel's slot count, so the low 11 bits hold the
// whole value (bits 11-16 of the 17-bit field are always 0) and the mask fits an immediate.
constexpr uint32_t e1_stream_count_mask = 0x7FFu;

template <uint32_t STREAM_ID>
FORCE_INLINE uint32_t e1_read_stream_count() {
    return NOC_STREAM_READ_REG(STREAM_ID, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX) & e1_stream_count_mask;
}

// On an intermesh router with VC0-to-VC1 crossover, receiver channel 0 forwards into the VC1 downstream connections, and
// kernel_main's VC0 adapters credit the downstream's VC1 sender channels too.
constexpr bool e1_vc0_crosses_to_vc1 =
#if defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)
    true;
#else
    false;
#endif

// Compile-time facts of the downstream router connections of VC `VC`, as kernel_main's adapters (RouterToRouterSender)
// are built for it: count (compact slots), slots per downstream ring, forwarding command buffers, and per compact slot
// the downstream sender channel's free-slots stream (its credit register) and this router's local copy of it.
template <size_t VC>
struct E1Downstreams {
    static constexpr size_t size = VC == 0 ? VC0_DOWNSTREAM_EDM_SIZE : VC1_DOWNSTREAM_EDM_SIZE;
    static constexpr size_t num_slots = VC == 0 ? DOWNSTREAM_SENDER_NUM_BUFFERS_VC0 : DOWNSTREAM_SENDER_NUM_BUFFERS_VC1;
    static constexpr uint8_t data_cmd_buf = receiver_channel_forwarding_data_cmd_buf_ids[VC];
    static constexpr uint8_t sync_cmd_buf = receiver_channel_forwarding_sync_cmd_buf_ids[VC];

    template <size_t COMPACT>
    static constexpr uint32_t local_credit_stream() {
        if constexpr (VC == 0) {
            return vc0_local_free_slots_stream_id<COMPACT>();
        } else {
            return vc1_local_free_slots_stream_id<COMPACT>();
        }
    }

    static constexpr std::array<uint32_t, MAX_DOWNSTREAM_EDM_COUNT> remote_credit_streams = [] {
        std::array<uint32_t, MAX_DOWNSTREAM_EDM_COUNT> ids{};
        for (uint32_t i = 0; i < MAX_DOWNSTREAM_EDM_COUNT; i++) {
#if defined(FABRIC_2D)
            ids[i] = (VC == 0 && !e1_vc0_crosses_to_vc1) ? get_vc0_downstream_sender_channel_free_slots_stream_id(i)
                                                          : get_vc1_downstream_sender_channel_free_slots_stream_id(i);
#else
            ids[i] = sender_channel_1_free_slots_stream_id;
#endif
        }
        return ids;
    }();
    static constexpr std::array<uint32_t, MAX_DOWNSTREAM_EDM_COUNT> local_credit_streams = [] {
        std::array<uint32_t, MAX_DOWNSTREAM_EDM_COUNT> ids{};
        for (uint32_t i = 0; i < MAX_DOWNSTREAM_EDM_COUNT; i++) {
#if defined(FABRIC_2D)
            ids[i] = VC == 0 ? vc0_local_free_slots_stream_id(i) : vc1_local_free_slots_stream_id(i);
#else
            ids[i] = vc0_local_free_slots_stream_id(0);
#endif
        }
        return ids;
    }();

    static_assert(size > 0 && size <= MAX_DOWNSTREAM_EDM_COUNT);
    static_assert(num_slots <= e1_stream_count_mask);
};

// Compile-time facts of receiver channel CH. It forwards into the downstream connections of its own VC, or of VC1 for
// VC0 with crossover.
template <size_t CH>
struct E1Rx {
    static constexpr size_t ds_vc = (CH == 0 && e1_vc0_crosses_to_vc1) ? 1 : CH;
    using DS = E1Downstreams<ds_vc>;
    static constexpr uint32_t num_slots = RECEIVER_NUM_BUFFERS_ARRAY[CH];
    static constexpr uint32_t slot_bytes = channel_buffer_size;
    static constexpr uint32_t first_slot = channel_allocs::template Entry<RECEIVER_TO_ENTRY_IDX[CH]>::base_address;
    static constexpr uint32_t pkts_stream = to_receiver_packets_sent_streams[CH];
    // Completion credits go back in batches of up to 4 slots on rings of 8 or more.
    static constexpr uint32_t batch_target = num_slots >= 8 ? 4 : 1;
    static constexpr size_t ds_size = DS::size;
    static constexpr uint8_t data_cmd_buf = DS::data_cmd_buf;
    static constexpr uint8_t sync_cmd_buf = DS::sync_cmd_buf;
    // Flushed atomics may be deferred past later packets (VC0 only; see can_defer_local_atomic_flush).
    static constexpr bool defers_flushed_atomics = CH == 0 && can_defer_local_atomic_flush<false>();
    // The VC1 receiver channel delivers, admits and forwards through shared out-of-line bodies (code size); receiver
    // channel 0 always uses the inline compile-time forms, so VC0 traffic costs the same with or without VC1.
    static constexpr bool uses_shared_bodies = CH == 1;

    template <size_t COMPACT>
    static constexpr uint32_t local_credit_stream() {
        return DS::template local_credit_stream<COMPACT>();
    }

    static_assert(num_slots > 0 && num_slots <= 255 && num_slots <= e1_stream_count_mask);
};

// Per-slot record of a receiver channel, one byte per slot: the sender channel the packet came from (src_ch_id, for the
// completion credit) in bits 0-5, and the command buffers whose NoC commands read the slot's data in bits 6-7:
// E1_SLOT_READ_LOCAL for a local write (local-write command buffer), E1_SLOT_READ_FWD for a forward (forwarding data
// command buffer). Local atomics and inline writes carry their values in command-buffer registers and do not read the
// slot; neither does a packet that is neither delivered nor forwarded. A record is written when its slot is retired
// (after the packet's NoC commands). Records are kept as words, so the four slots of a completion batch are one word
// load (two when the batch straddles a word).
constexpr uint32_t E1_SLOT_SRC_MASK = 0x3F;
constexpr uint32_t E1_SLOT_READ_LOCAL = 0x40;
constexpr uint32_t E1_SLOT_READ_FWD = 0x80;
constexpr uint32_t E1_SLOT_READS = E1_SLOT_READ_LOCAL | E1_SLOT_READ_FWD;
static_assert(MAX_NUM_SENDER_CHANNELS <= E1_SLOT_SRC_MASK + 1);
// Local send types whose NoC commands read the slot: write, fused write + atomic, scatter, sparse multicast.
constexpr uint32_t E1_LOCAL_TYPES_READING_SLOT = (1u << tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE) |
                                                 (1u << tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) |
                                                 (1u << tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE) |
                                                 (1u << tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE);

static_assert(tt::tt_fabric::NocSendType::NOC_SEND_TYPE_LAST < 32);

FORCE_INLINE uint32_t e1_local_read_bits(tt::tt_fabric::NocSendType noc_send_type) {
    return ((E1_LOCAL_TYPES_READING_SLOT >> (static_cast<uint32_t>(noc_send_type) & 31u)) & 0x1u) * E1_SLOT_READ_LOCAL;
}

// Slot `base + n` of a ring of N slots (n < N). Unlike wrap_increment_n, also right for N = 2.
template <uint32_t N>
FORCE_INLINE uint8_t e1_slot_add(uint32_t base, uint32_t n) {
    const uint32_t i = base + n;
    if constexpr ((N & (N - 1)) == 0) {
        return static_cast<uint8_t>(i & (N - 1));
    } else {
        return static_cast<uint8_t>(i >= N ? i - N : i);
    }
}

template <uint32_t N>
struct E1SlotRecords {
    static constexpr uint32_t num_words = (N + 3) / 4;
    std::array<uint32_t, num_words> words;

    FORCE_INLINE void set(uint8_t idx, uint32_t record) {
        reinterpret_cast<uint8_t*>(words.data())[idx] = static_cast<uint8_t>(record);
    }
    FORCE_INLINE uint32_t get(uint8_t idx) const { return reinterpret_cast<const uint8_t*>(words.data())[idx]; }
    // Records of slots base .. base + 3 (ring order), slot base in the low byte.
    FORCE_INLINE uint32_t four_from(uint32_t base) const {
        if constexpr (N % 4 == 0) {
            const uint32_t w = base >> 2;
            const uint32_t shift = (base & 3) * 8;
            uint32_t four = words[w];
            if (shift != 0) {
                const uint32_t next = w + 1 == num_words ? 0 : w + 1;
                four = (four >> shift) | (words[next] << (32 - shift));
            }
            return four;
        } else {
            uint32_t four = 0;
            uint8_t idx = static_cast<uint8_t>(base);
            for (uint32_t k = 0; k < 4; k++) {
                four |= get(idx) << (8 * k);
                idx = tt::tt_fabric::wrap_increment<N>(idx);
            }
            return four;
        }
    }
};

// Loop-carried state of one receiver channel.
struct E1RxState {
    uint32_t avail;           // packets known to be in the ring and not yet retired (lower bound of the register)
    uint32_t slot_addr;       // header address of the next slot to retire
    uint32_t uncompleted;     // slots retired (NoC commands issued) but not yet credited back to the far sender
    uint32_t idle_polls;      // consecutive passes without progress while a partial batch waits (2D)
    uint32_t pending_atomic;  // deferred flushed atomic (encoding: drain_pending_atomic_flush); VC0, 2D only
    uint8_t idx;              // buffer index of the next slot to retire
    // STABILITY_PROBE_STATE
};

template <size_t CH>
FORCE_INLINE void e1_rx_state_init(E1RxState& st) {
    st.avail = 0;
    st.slot_addr = E1Rx<CH>::first_slot;
    st.uncompleted = 0;
    st.idle_polls = 0;
    st.pending_atomic = 0;
    st.idx = 0;
}

// One downstream router connection as E1 forwards to it: the next slot in the downstream sender channel's ring, the
// ring bounds, the downstream's NoC coordinate and credit register (stateless sends), the local copy of its free slots
// (read and update addresses, for runtime-indexed slots), and a count of free slots known from the last read of that
// copy. E1 sets these connections up itself (e1_init_downstreams, e1_open_downstreams, e1_setup_downstream_cmd_bufs:
// the same writes, in the same order, as kernel_main's RouterToRouterSender adapters) and forwards only through them.
struct E1DownstreamCursor {
    uint32_t slot_addr;
    uint32_t first_slot_addr;
    uint32_t last_slot_addr;
    uint32_t free_slots;
    uint32_t noc_coord;
    uint32_t local_credit_read_addr;
    uint32_t local_credit_update_addr;
    uint64_t credit_noc_addr;
};

// Has the downstream a free slot? Re-reads the local copy register only when the known count is used up.
template <uint32_t LOCAL_CREDIT_STREAM>
FORCE_INLINE bool e1_downstream_has_space(E1DownstreamCursor& c) {
    if (c.free_slots == 0) {
        c.free_slots = e1_read_stream_count<LOCAL_CREDIT_STREAM>();
    }
    return c.free_slots != 0;
}

FORCE_INLINE bool e1_downstream_has_space_rt(E1DownstreamCursor& c) {
    if (c.free_slots == 0) {
        c.free_slots = *reinterpret_cast<volatile uint32_t*>(c.local_credit_read_addr) & e1_stream_count_mask;
    }
    return c.free_slots != 0;
}

// The forward: the whole packet into the downstream's next slot (tagged with the slot's trid), the downstream's
// free-slots register -1 (its doorbell), and the local copy -1. The same NoC commands, in the same order, as
// RouterToRouterSender::send_payload_non_blocking_from_address_with_trid without deadlock avoidance.
template <bool STATEFUL, uint8_t DATA_CMD_BUF, uint8_t SYNC_CMD_BUF>
FORCE_INLINE void e1_forward_payload(E1DownstreamCursor& c, uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
    constexpr uint8_t noc = tt::tt_fabric::edm_to_downstream_noc;
    const uint32_t dst_addr = c.slot_addr;
    send_chunk_from_address_with_trid<STATEFUL>(src_addr, 1, size_bytes, c.noc_coord, dst_addr, trid, noc, DATA_CMD_BUF);
    c.slot_addr = dst_addr == c.last_slot_addr ? c.first_slot_addr : dst_addr + channel_buffer_size;
    if constexpr (STATEFUL) {
        noc_inline_dw_write_with_state<false, false, true, false, false, InlineWriteDst::REG>(0, 0, SYNC_CMD_BUF, noc);
    } else {
        noc_inline_dw_write<InlineWriteDst::REG>(
            c.credit_noc_addr, pack_value_for_inc_on_write_stream_reg_write(-1), 0xf, noc);
    }
    c.free_slots--;
}

#if defined(FABRIC_2D_VC1_ACTIVE)
// The VC1 receiver channel (channel 1) delivers, admits and forwards through one out-of-line body each, which walk the
// downstream slots an action uses at run time, so the second receiver channel adds little code (VC1 routers must fit
// both ERISCs in the kernel config buffer). Receiver channel 0 keeps the inline compile-time forms. VC1 forwards are
// stateless.
template <bool UNUSED = true>
__attribute__((noinline)) bool e1_admit_slots_shared(uint32_t slots, E1DownstreamCursor* ds) {
    bool ok = true;
    for (uint32_t i = 0; slots != 0; i++, slots >>= 1) {
        if (slots & 0x1) {
            ok = e1_downstream_has_space_rt(ds[i]) && ok;
        }
    }
    return ok;
}

template <uint8_t DATA_CMD_BUF>
__attribute__((noinline)) void e1_forward_slots_shared(
    uint32_t slots, E1DownstreamCursor* ds, uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
    for (uint32_t i = 0; slots != 0; i++, slots >>= 1) {
        if (slots & 0x1) {
            e1_forward_payload<false, DATA_CMD_BUF, DATA_CMD_BUF>(ds[i], src_addr, size_bytes, trid);
            *reinterpret_cast<volatile uint32_t*>(ds[i].local_credit_update_addr) =
                pack_value_for_inc_on_write_stream_reg_write(-1);
        }
    }
}

template <bool UNUSED = true>
__attribute__((noinline)) void e1_local_delivery_shared(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
    uint16_t payload_size_bytes,
    tt::tt_fabric::NocSendType noc_send_type,
    uint8_t transaction_id,
    uint8_t rx_channel_id) {
    execute_chip_unicast_to_local_chip_impl(
        packet_start, payload_size_bytes, noc_send_type, transaction_id, rx_channel_id, nullptr);
}
#endif

// Forward to the downstream in compact slot COMPACT of channel CH (compile-time slot; not used with VC1).
template <size_t CH, size_t COMPACT, bool STATEFUL>
FORCE_INLINE void e1_forward_to(E1DownstreamCursor& c, uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
    using C = E1Rx<CH>;
    e1_forward_payload<STATEFUL, C::data_cmd_buf, C::sync_cmd_buf>(c, src_addr, size_bytes, trid);
    increment_local_update_ptr_val<C::template local_credit_stream<COMPACT>()>(-1);
}

// Forward to a downstream picked at run time (the intermesh exit of an edge router); always stateless.
template <size_t CH>
FORCE_INLINE void e1_forward_to_rt(
    std::array<E1DownstreamCursor, E1Rx<CH>::ds_size>& ds,
    uint32_t compact_index,
    uint32_t src_addr,
    uint32_t size_bytes,
    uint8_t trid) {
    using C = E1Rx<CH>;
#if defined(FABRIC_2D_VC1_ACTIVE)
    if constexpr (C::uses_shared_bodies) {
        e1_forward_slots_shared<C::data_cmd_buf>(1u << compact_index, ds.data(), src_addr, size_bytes, trid);
        return;
    }
#endif
    E1DownstreamCursor& c = ds[compact_index];
    e1_forward_payload<false, C::data_cmd_buf, C::sync_cmd_buf>(c, src_addr, size_bytes, trid);
    *reinterpret_cast<volatile uint32_t*>(c.local_credit_update_addr) = pack_value_for_inc_on_write_stream_reg_write(-1);
}

// One local write of the scatter arm without the software write counters (noc_nonposted_writes_num_issued/acked):
// noc_async_write_one_packet_with_trid's command, with the counter update left to the caller, which adds all of the
// packet's writes at once (as the DRISC scatter path does). Nothing reads the counters between the writes.
FORCE_INLINE void e1_local_write_uncounted(uint32_t src_addr, uint64_t dst_noc_addr, uint32_t size, uint32_t trid) {
    constexpr uint8_t cmd_buf = tt::tt_fabric::local_chip_data_cmd_buf;
    constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
    WAYPOINT("NAWW");
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_TRID, src_addr, dst_noc_addr, size, -1, false, noc);
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(noc, dst_noc_addr, src_addr, size);
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    ncrisc_noc_fast_write<noc_mode, /*use_trid=*/true, /*update_counter=*/false>(
        noc,
        cmd_buf,
        src_addr,
        dst_noc_addr,
        size,
        tt::tt_fabric::forward_and_local_write_noc_vc,
        false,  // mcast
        false,  // linked
        1,      // num_dests
        true,   // multicast_path_reserve
        false,  // posted
        trid);
    WAYPOINT("NWPD");
}

// Local delivery for receiver channel CH. A plain write (the common case), a scatter write, an atomic increment and a
// fused write + atomic increment are issued directly, with the same NoC commands as the matching arm of
// execute_chip_unicast_to_local_chip_impl and without its send-type dispatch (sparse test, range test, jump table). A
// scatter write updates the software write counters once for all of its writes, before a final semaphore increment,
// whose flush reads them. Every other send type goes through execute_chip_unicast_to_local_chip_impl, unchanged. Adds
// the slot's read class to `record`. The VC1 receiver channel keeps the shared out-of-line body (VC1 code size).
template <size_t CH>
FORCE_INLINE void e1_local_delivery(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
    uint16_t payload_size_bytes,
    tt::tt_fabric::NocSendType noc_send_type,
    uint8_t trid,
    [[maybe_unused]] uint32_t* pending_atomic_flush,
    uint32_t& record) {
#if defined(FABRIC_2D_VC1_ACTIVE)
    if constexpr (E1Rx<CH>::uses_shared_bodies) {
        record |= e1_local_read_bits(noc_send_type);
        e1_local_delivery_shared(packet_start, payload_size_bytes, noc_send_type, trid, CH);
        return;
    }
#endif
    using tt::tt_fabric::NocSendType;
    const uint32_t payload_start_address = reinterpret_cast<size_t>(packet_start) + sizeof(PACKET_HEADER_TYPE);
    if (noc_send_type == NocSendType::NOC_UNICAST_WRITE) {
        record |= E1_SLOT_READ_LOCAL;
        channel_trimming_usage_recorder.set_noc_send_type_used(CH, noc_send_type);
        noc_async_write_one_packet_with_trid</*update_counter=*/false, /*posted=*/false>(
            payload_start_address,
            packet_start->command_fields.unicast_write.noc_address,
            payload_size_bytes,
            trid,
            tt::tt_fabric::local_chip_data_cmd_buf,
            tt::tt_fabric::edm_to_local_chip_noc,
            tt::tt_fabric::forward_and_local_write_noc_vc);
    } else if (noc_send_type == NocSendType::NOC_UNICAST_SCATTER_WRITE) {
        record |= E1_SLOT_READ_LOCAL;
        channel_trimming_usage_recorder.set_noc_send_type_used(CH, noc_send_type);
        constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
        using ChunkEncoding = tt::tt_fabric::NocScatterWriteChunkEncoding;
        const auto& scatter = packet_start->command_fields.unicast_scatter_write;
        const uint8_t chunk_count = scatter.chunk_count;
        uint8_t packet_encoding = scatter.chunk_encoding;
        // Up to three unicast writes, then the last chunk: a unicast write of the rest of the payload, or a semaphore
        // increment (its value carried in the chunk size).
        uint32_t offset = scatter.chunk_size[0];
        uint32_t num_write_chunks = 1;
        e1_local_write_uncounted(payload_start_address, scatter.noc_address[0], offset, trid);
        shift_to_next_chunk(packet_encoding);
        if (chunk_count > 2) {
            const uint32_t chunk_size_1 = scatter.chunk_size[1];
            e1_local_write_uncounted(payload_start_address + offset, scatter.noc_address[1], chunk_size_1, trid);
            offset += chunk_size_1;
            shift_to_next_chunk(packet_encoding);
            num_write_chunks++;
            if (chunk_count == 4) {
                const uint32_t chunk_size_2 = scatter.chunk_size[2];
                e1_local_write_uncounted(payload_start_address + offset, scatter.noc_address[2], chunk_size_2, trid);
                offset += chunk_size_2;
                shift_to_next_chunk(packet_encoding);
                num_write_chunks++;
            }
        }
        constexpr uint8_t CHUNK_ENCODING_MASK = 0b11;
        const ChunkEncoding chunk_encoding = static_cast<ChunkEncoding>(packet_encoding & CHUNK_ENCODING_MASK);
        const uint8_t last_chunk_index = chunk_count - 1;
        const uint64_t final_destination_noc_address = scatter.noc_address[last_chunk_index];
        if (chunk_encoding == ChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE) {
            e1_local_write_uncounted(
                payload_start_address + offset,
                final_destination_noc_address,
                static_cast<uint16_t>(payload_size_bytes - offset),
                trid);
            num_write_chunks++;
        }
        noc_nonposted_writes_num_issued[noc] += num_write_chunks;
        noc_nonposted_writes_acked[noc] += num_write_chunks;
        if (chunk_encoding != ChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE) {
            if (chunk_encoding != ChunkEncoding::CHUNK_ENCODING_NOP) {
                execute_local_semaphore_inc(
                    final_destination_noc_address,
                    scatter.chunk_size[last_chunk_index],
                    chunk_encoding == ChunkEncoding::CHUNK_ENCODING_SEMINC_FLUSH
                        ? static_cast<uint8_t>(
                              tt::tt_fabric::NocFlushType::FLUSH_STRICT + ((packet_encoding >> 2) & 0x1u))
                        : static_cast<uint8_t>(tt::tt_fabric::NocFlushType::FLUSH_NONE),
                    CH,
                    pending_atomic_flush,
                    trid);
            } else {
                ASSERT(false);
            }
        }
    } else if (noc_send_type == NocSendType::NOC_UNICAST_ATOMIC_INC) {
        channel_trimming_usage_recorder.set_noc_send_type_used(CH, noc_send_type);
        execute_local_semaphore_inc(
            packet_start->command_fields.unicast_seminc.noc_address,
            packet_start->command_fields.unicast_seminc.val,
            packet_start->command_fields.unicast_seminc.flush,
            CH,
            pending_atomic_flush,
            static_cast<uint8_t>((trid - 1u) & 0xFu));
    } else if (noc_send_type == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) {
        record |= E1_SLOT_READ_LOCAL;
        channel_trimming_usage_recorder.set_noc_send_type_used(CH, noc_send_type);
        noc_async_write_one_packet_with_trid</*update_counter=*/false, /*posted=*/false>(
            payload_start_address,
            packet_start->command_fields.unicast_seminc_fused.noc_address,
            payload_size_bytes,
            trid,
            tt::tt_fabric::local_chip_data_cmd_buf,
            tt::tt_fabric::edm_to_local_chip_noc,
            tt::tt_fabric::forward_and_local_write_noc_vc);
        execute_local_semaphore_inc(
            packet_start->command_fields.unicast_seminc_fused.semaphore_noc_address,
            packet_start->command_fields.unicast_seminc_fused.val,
            packet_start->command_fields.unicast_seminc_fused.flush,
            CH,
            pending_atomic_flush,
            trid);
    } else {
        record |= e1_local_read_bits(noc_send_type);
        execute_chip_unicast_to_local_chip_impl(
            packet_start, payload_size_bytes, noc_send_type, trid, CH, pending_atomic_flush);
    }
}

#if defined(FABRIC_2D)
// Does the action use compact downstream slot I of channel CH (a LIVE direction whose action bit is set)? Bits outside
// LIVE are ignored, as in admit_2d_dispatch / forward_2d_dispatch.
template <size_t CH, size_t I>
FORCE_INLINE bool e1_action_uses(uint8_t action) {
    constexpr uint8_t LIVE = live_eth_mask_2d<E1Rx<CH>::ds_size>();
    if constexpr (((LIVE >> I) & 1) != 0) {
        constexpr auto dirs = Routing2DCodec::fwd_dirs<static_cast<eth_chan_directions>(my_direction)>();
        return (action & Routing2DCodec::action_bit(dirs[I])) != 0;
    } else {
        return false;
    }
}

// Compact downstream slots the action uses (bit i = compact slot i).
template <size_t CH>
FORCE_INLINE uint32_t e1_action_slots(uint8_t action) {
    return (e1_action_uses<CH, 0>(action) ? 0x1u : 0u) | (e1_action_uses<CH, 1>(action) ? 0x2u : 0u) |
           (e1_action_uses<CH, 2>(action) ? 0x4u : 0u) | (e1_action_uses<CH, 3>(action) ? 0x8u : 0u);
}

// Known-free count of compact slot I is used up (only for LIVE slots the action uses).
template <size_t CH, size_t I>
FORCE_INLINE bool e1_needs_reload(uint8_t action, std::array<E1DownstreamCursor, E1Rx<CH>::ds_size>& ds) {
    constexpr uint8_t LIVE = live_eth_mask_2d<E1Rx<CH>::ds_size>();
    if constexpr (((LIVE >> I) & 1) != 0) {
        return e1_action_uses<CH, I>(action) && ds[I].free_slots == 0;
    } else {
        return false;
    }
}

template <size_t CH, size_t I>
FORCE_INLINE uint32_t e1_read_live_credit() {
    constexpr uint8_t LIVE = live_eth_mask_2d<E1Rx<CH>::ds_size>();
    if constexpr (((LIVE >> I) & 1) != 0) {
        return NOC_STREAM_READ_REG(
            E1Rx<CH>::template local_credit_stream<I>(), STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
    } else {
        return 0;
    }
}

template <size_t CH, size_t I>
FORCE_INLINE void e1_set_live_credit(std::array<E1DownstreamCursor, E1Rx<CH>::ds_size>& ds, uint32_t raw) {
    constexpr uint8_t LIVE = live_eth_mask_2d<E1Rx<CH>::ds_size>();
    if constexpr (((LIVE >> I) & 1) != 0) {
        ds[I].free_slots = raw & e1_stream_count_mask;
    }
}

template <size_t CH, size_t I>
FORCE_INLINE bool e1_has_credit_if_used(uint8_t action, std::array<E1DownstreamCursor, E1Rx<CH>::ds_size>& ds) {
    constexpr uint8_t LIVE = live_eth_mask_2d<E1Rx<CH>::ds_size>();
    if constexpr (((LIVE >> I) & 1) != 0) {
        return !e1_action_uses<CH, I>(action) || ds[I].free_slots != 0;
    } else {
        return true;
    }
}

// Admission for the forwarding part of a 2D action: every LIVE direction the action uses needs a free slot downstream.
// Local delivery needs nothing (no UDM here). The known-free counts are tested first; only when a needed one is used
// up are the live local copy registers re-read, as one group.
template <size_t CH>
FORCE_INLINE bool e1_admit_2d(uint8_t action, std::array<E1DownstreamCursor, E1Rx<CH>::ds_size>& ds) {
    using C = E1Rx<CH>;
    constexpr uint8_t LIVE = live_eth_mask_2d<C::ds_size>();
    if constexpr (LIVE == 0) {
        return true;
    } else if constexpr (LIVE == 0b0001) {
        if (e1_action_uses<CH, 0>(action)) {
            return e1_downstream_has_space<C::template local_credit_stream<0>()>(ds[0]);
        }
        return true;
    } else {
        if ((action & Routing2DCodec::ACTION_ETH_MASK) == 0) {
            return true;
        }
        const bool reload = e1_needs_reload<CH, 0>(action, ds) || e1_needs_reload<CH, 1>(action, ds) ||
                            e1_needs_reload<CH, 2>(action, ds) || e1_needs_reload<CH, 3>(action, ds);
        if (reload) {
            std::atomic_signal_fence(std::memory_order_seq_cst);
            const uint32_t raw0 = e1_read_live_credit<CH, 0>();
            const uint32_t raw1 = e1_read_live_credit<CH, 1>();
            const uint32_t raw2 = e1_read_live_credit<CH, 2>();
            const uint32_t raw3 = e1_read_live_credit<CH, 3>();
            std::atomic_signal_fence(std::memory_order_seq_cst);
            e1_set_live_credit<CH, 0>(ds, raw0);
            e1_set_live_credit<CH, 1>(ds, raw1);
            e1_set_live_credit<CH, 2>(ds, raw2);
            e1_set_live_credit<CH, 3>(ds, raw3);
        }
        return e1_has_credit_if_used<CH, 0>(action, ds) && e1_has_credit_if_used<CH, 1>(action, ds) &&
               e1_has_credit_if_used<CH, 2>(action, ds) && e1_has_credit_if_used<CH, 3>(action, ds);
    }
}

template <size_t CH, size_t I>
FORCE_INLINE uint8_t e1_forward_if_used(
    uint8_t action,
    uint32_t src_addr,
    uint32_t size_bytes,
    uint8_t trid,
    std::array<E1DownstreamCursor, E1Rx<CH>::ds_size>& ds,
    uint32_t& record) {
    constexpr uint8_t LIVE = live_eth_mask_2d<E1Rx<CH>::ds_size>();
    if constexpr (((LIVE >> I) & 1) != 0) {
        if (e1_action_uses<CH, I>(action)) {
            constexpr bool stateful = use_2d_single_live_stateful_noc<E1Rx<CH>::ds_size>();
            e1_forward_to<CH, I, stateful>(ds[I], src_addr, size_bytes, trid);
            record |= E1_SLOT_READ_FWD;
            return static_cast<uint8_t>(1u << I);
        }
    }
    return 0;
}

// One unmodified full-packet copy per LIVE direction the action uses, in compact slot order (local delivery is the
// caller's). Returns the mask of compact slots forwarded to, for channel-trimming capture; adds the forward read class to
// `record` when it forwards.
template <size_t CH>
FORCE_INLINE uint8_t e1_forward_2d(
    uint8_t action,
    uint32_t src_addr,
    uint32_t size_bytes,
    uint8_t trid,
    std::array<E1DownstreamCursor, E1Rx<CH>::ds_size>& ds,
    uint32_t& record) {
    uint8_t forwarded_slot_mask = e1_forward_if_used<CH, 0>(action, src_addr, size_bytes, trid, ds, record);
    forwarded_slot_mask |= e1_forward_if_used<CH, 1>(action, src_addr, size_bytes, trid, ds, record);
    forwarded_slot_mask |= e1_forward_if_used<CH, 2>(action, src_addr, size_bytes, trid, ds, record);
    forwarded_slot_mask |= e1_forward_if_used<CH, 3>(action, src_addr, size_bytes, trid, ds, record);
    return forwarded_slot_mask;
}
#endif

// Completion credits. While packets keep arriving, a batch goes out once more than batch_target slots wait (so the
// newest slot, which may still carry a deferred atomic, is never in it). On a pass without progress, a full batch goes
// out at once and a partial one after 4 such passes in 2D (at once in 1D).
//
// A batch is credited only when no NoC command still reads any of its slots, and the receiver TX queue is free. The
// check relies on one ordering fact, the same one v3a's last-slot check relies on: the commands of one command buffer
// leave the NIU in issue order, so when a slot's transaction ID has no outgoing write, no earlier slot has one on that
// command buffer either. From the batch's slot records (one word): the newest slot that reads its data proves every
// older slot on the command buffers it uses; if an older slot reads on the other command buffer, the newest such slot
// is read as well (two status reads, as one group). A batch whose slots all read the same way (all local writes, all
// forwards, all multicast) is one read of its newest slot, as v3a's; a batch with no slot read (atomics, inline
// writes) needs none. The TX queue is read after the trids, as in v3a.
template <size_t CH, typename WriteTridTracker, typename CreditSender>
FORCE_INLINE void e1_rx_completions(
    E1RxState& st,
    E1SlotRecords<E1Rx<CH>::num_slots>& records,
    WriteTridTracker& trid_tracker,
    CreditSender& credit_sender,
    bool progress) {
    using C = E1Rx<CH>;
    constexpr uint32_t T = C::batch_target;
    constexpr uint32_t N = C::num_slots;
    static_assert(T <= 4);
    const uint32_t uncompleted = st.uncompleted;
#if defined(FABRIC_2D)
    bool should_check;
    if (progress) {
        st.idle_polls = 0;
        should_check = uncompleted > T;
    } else {
        should_check = uncompleted >= T || (uncompleted != 0 && ++st.idle_polls >= 4);
    }
#else
    const bool should_check = uncompleted > T || (!progress && uncompleted != 0);
#endif
    if (!should_check) {
        return;
    }
    const uint32_t batch = uncompleted > T ? T : uncompleted;
    // Oldest uncredited slot: `uncompleted` slots behind the next one to retire.
    uint32_t base = static_cast<uint32_t>(st.idx) + N - uncompleted;
    if constexpr ((N & (N - 1)) == 0) {
        base &= N - 1;
    } else {
        base = base >= N ? base - N : base;
    }
    const uint8_t base_idx = static_cast<uint8_t>(base);

    // The batch's records, oldest in the low byte; bytes past the batch cleared.
    uint32_t recs = records.four_from(base);
    if (batch < 4) {
        recs &= (1u << (8 * batch)) - 1;
    }
    const uint32_t reads = recs & (E1_SLOT_READS * 0x01010101u);
    if (reads != 0) {
        constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
        volatile uint32_t* const outgoing = reinterpret_cast<volatile uint32_t*>(
            (static_cast<uint32_t>(noc) << NOC_INSTANCE_OFFSET_BIT) + NOC_STATUS(NIU_MST_WRITE_REQS_OUTGOING_ID(0)));
        // Newest slot that reads its data, and the reads it does not cover (older slots, other command buffer).
        const uint32_t newest = (31 - static_cast<uint32_t>(__builtin_clz(reads))) >> 3;
        const uint32_t covered = (recs >> (8 * newest)) & E1_SLOT_READS;
        const uint32_t other = reads & ~(covered * 0x01010101u);
        const uint8_t newest_idx = e1_slot_add<N>(base, newest);
        const uint32_t newest_trid = trid_tracker.get_buffer_slot_trid(tt::tt_fabric::BufferIndex{newest_idx});
        if (other == 0) {
            if (outgoing[newest_trid] != 0) {
                return;
            }
        } else {
            const uint32_t older = (31 - static_cast<uint32_t>(__builtin_clz(other))) >> 3;
            const uint8_t older_idx = e1_slot_add<N>(base, older);
            const uint32_t older_trid = trid_tracker.get_buffer_slot_trid(tt::tt_fabric::BufferIndex{older_idx});
            std::atomic_signal_fence(std::memory_order_seq_cst);
            const uint32_t newest_out = outgoing[newest_trid];
            const uint32_t older_out = outgoing[older_trid];
            std::atomic_signal_fence(std::memory_order_seq_cst);
            if ((newest_out | older_out) != 0) {
                return;
            }
        }
    }
    if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
        if (internal_::eth_txq_is_busy(receiver_txq_id)) {
            return;
        }
    }

#if defined(FABRIC_2D)
    st.idle_polls = 0;
#endif
    // A full batch from one sender channel (the usual case) is one credit of 4 to it; otherwise one credit per slot.
    // The counters end the same either way, and the frame is sent only below.
    const uint32_t diff = recs ^ (recs >> 8);  // byte k < 3: record k ^ record k + 1
    if (batch == 4 && (diff & (E1_SLOT_SRC_MASK * 0x00010101u)) == 0) {
        credit_sender.accumulate_completion_credit(static_cast<uint8_t>(recs & E1_SLOT_SRC_MASK), 4);
    } else {
        for (uint32_t k = 0; k < batch; k++) {
            credit_sender.accumulate_completion_credit(static_cast<uint8_t>((recs >> (8 * k)) & E1_SLOT_SRC_MASK), 1);
        }
    }
    if constexpr (!(WriteTridTracker::BOTH_PARAMS_ARE_EQUAL || WriteTridTracker::BOTH_PARAMS_ARE_POW2)) {
        uint8_t idx = base_idx;
        for (uint32_t k = 0; k < batch; k++) {
            trid_tracker.clear_trid_at_buffer_slot(tt::tt_fabric::BufferIndex{idx});
            idx = tt::tt_fabric::wrap_increment<N>(idx);
        }
    }
    if constexpr (ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
        while (internal_::eth_txq_is_busy(receiver_txq_id)) {
        }
    }
    credit_sender.flush_completion_credits();
    st.uncompleted = uncompleted - batch;
}

// Retire the slot just handled: packets-received register -1 (after every NoC command of the packet), next slot.
template <size_t CH>
FORCE_INLINE void e1_retire(E1RxState& st) {
    using C = E1Rx<CH>;
    st.avail--;
    increment_local_update_ptr_val<C::pkts_stream>(-1);
    st.uncompleted++;
    const uint8_t next_idx = tt::tt_fabric::wrap_increment<C::num_slots>(st.idx);
    st.idx = next_idx;
    st.slot_addr = next_idx == 0 ? C::first_slot : st.slot_addr + C::slot_bytes;
}

// One pass over receiver channel CH: at most one packet, then the deferred-atomic idle drain and completions.
template <size_t CH, typename WriteTridTracker, typename CreditSender, typename LocalTelemetryT>
FORCE_INLINE bool e1_rx_step(
    E1RxState& st,
    E1SlotRecords<E1Rx<CH>::num_slots>& records,
    std::array<E1DownstreamCursor, E1Rx<CH>::ds_size>& ds,
    WriteTridTracker& trid_tracker,
    CreditSender& credit_sender,
    [[maybe_unused]] const tt::tt_fabric::routing_l1_info_t& routing_table,
    [[maybe_unused]] uint16_t my_mesh_id,
    [[maybe_unused]] uint8_t my_mesh_coord_y,
    [[maybe_unused]] uint8_t my_mesh_coord_x,
    [[maybe_unused]] LocalTelemetryT& local_fabric_telemetry) {
    using C = E1Rx<CH>;
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    if (st.avail == 0) {
        st.avail = e1_read_stream_count<C::pkts_stream>();
    }

    NamedProfiler<
        CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD,
        code_profiling_enabled_timers_bitfield,
        code_profiling_buffer_base_addr>
        receiver_forward_timer;
    receiver_forward_timer.set_should_dump(st.avail != 0);
    receiver_forward_timer.open();

    bool progress = false;
    if (st.avail != 0) {
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        tt_l1_ptr PACKET_HEADER_TYPE* const packet_header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(st.slot_addr);
#if defined(FABRIC_2D)
        const auto hdr_w40 = PackedHeaderWord40::load(packet_header);
        uint32_t record = hdr_w40.src_ch_id;  // the slot's record (E1SlotRecords), stored when it is retired
        // Same decode as run_receiver_channel_step_impl: landing intercept (intermesh ingress) -> action decode ->
        // intermesh-exit intercept (edge routers) -> admission on the action bits.
        uint8_t action;
        bool local_deliver = false;
        bool intermesh_egress = false;
        uint8_t intermesh_egress_index = 0;
        [[maybe_unused]] uint32_t fwd_slots = 0;
        bool admitted;
        if constexpr (receiver_channel_is_intermesh_ingress[CH]) {
            fabric_set_2d_intermesh_landing_route(packet_header, routing_table, MESH_Y_SIZE, MESH_X_SIZE);
            action = Routing2DCodec::decode_action_y_first(
                packet_header->route_buffer, my_mesh_coord_y, my_mesh_coord_x, MESH_Y_SIZE);
        } else {
            action = Routing2DCodec::decode_action<static_cast<eth_chan_directions>(my_direction)>(
                packet_header->route_buffer, my_mesh_coord_y, my_mesh_coord_x, MESH_Y_SIZE);
        }
        bool intermesh_exit = false;
        if constexpr (is_intramesh_router_on_edge) {
            intermesh_exit =
                Routing2DCodec::action_is_intermesh_exit(action, packet_header->dst_start_mesh_id, my_mesh_id);
        }
        if (intermesh_exit) {
            const auto boundary_dir = static_cast<eth_chan_directions>(
                routing_table.inter_mesh_direction_table.get_original_direction(packet_header->dst_start_mesh_id));
            const bool boundary_dir_is_addressable =
                boundary_dir < eth_chan_directions::COUNT &&
                boundary_dir != static_cast<eth_chan_directions>(my_direction) &&
                get_downstream_edm_interface_index(boundary_dir) < C::ds_size;
            if (!boundary_dir_is_addressable) {
                ASSERT(false);
                admitted = false;
            } else {
                intermesh_egress = true;
                intermesh_egress_index = static_cast<uint8_t>(get_downstream_edm_interface_index(boundary_dir));
#if defined(FABRIC_2D_VC1_ACTIVE)
                if constexpr (C::uses_shared_bodies) {
                    admitted = e1_admit_slots_shared(1u << intermesh_egress_index, ds.data());
                } else
#endif
                {
                    admitted = e1_downstream_has_space_rt(ds[intermesh_egress_index]);
                }
            }
        } else {
            local_deliver = (action & Routing2DCodec::ACTION_LOCAL_DELIVER) != 0;
#if defined(FABRIC_2D_VC1_ACTIVE)
            if constexpr (C::uses_shared_bodies) {
                fwd_slots = e1_action_slots<CH>(action);
                admitted = fwd_slots == 0 || e1_admit_slots_shared(fwd_slots, ds.data());
            } else
#endif
            {
                admitted = e1_admit_2d<CH>(action, ds);
            }
        }
#else
        const ROUTING_FIELDS_TYPE cached_routing_fields = packet_header->routing_fields;
        uint32_t record = packet_header->src_ch_id;  // the slot's record (E1SlotRecords), stored when it is retired
        const uint32_t routing = cached_routing_fields.value & LowLatencyFields::FIELD_MASK;
        const bool admitted = routing == LowLatencyFields::WRITE_ONLY ||
                              e1_downstream_has_space<C::template local_credit_stream<0>()>(ds[0]);
#endif
        if (admitted) {
            progress = true;
            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                update_bw_counters(packet_header, local_fabric_telemetry);
            }
            channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(CH);
            const uint8_t trid = trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(
                tt::tt_fabric::BufferIndex{st.idx});
#if defined(FABRIC_2D)
            if constexpr (C::defers_flushed_atomics) {
                // A deferred flushed atomic is pending. 1: strict (waits for every earlier write on the NoC).
                // Relaxed: 0x10 | the trid of its own packet, then 0x100 | (trid_b << 4) | trid_a once the one or two
                // trids it still waits for are known. Resolved here, before this packet issues any NoC command.
                uint32_t& pending_atomic_flush = st.pending_atomic;
                if (pending_atomic_flush) {
                    if (pending_atomic_flush == 1u) {
                        complete_pending_atomic_flush(CH);
                        pending_atomic_flush = 0;
                    } else {
                        constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
                        const uint32_t trid_s = pending_atomic_flush & 0xFu;
                        const bool can_bypass =
                            hdr_w40.noc_send_type == tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE &&
                            (use_2d_single_live_stateful_noc<C::ds_size>() ||
                             action == Routing2DCodec::ACTION_LOCAL_DELIVER) &&
                            !intermesh_egress && ((trid_s - trid) & 0xFu) > 1u;
                        if (!can_bypass) {
                            drain_pending_atomic_flush(CH, pending_atomic_flush);
                        } else if (pending_atomic_flush < 0x100u) {
                            // First write past the atomic: nothing has been issued since its packet, so every
                            // outstanding write is older. Read global outstanding first, then trid_s and
                            // trid_p = (trid_s - 1) & 0xF. Since counts only fall, g_out == s_out + p_out proves no
                            // write is outstanding on any other trid (regardless of ack order).
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
                    }
                }
            }
            const uint32_t src_addr = reinterpret_cast<uint32_t>(packet_header);
            const uint32_t size_bytes = hdr_w40.payload_size_bytes + sizeof(PACKET_HEADER_TYPE);
            if (intermesh_egress) {
                // The exit chip's only output is the INTERMESH egress, sent as-is.
                e1_forward_to_rt<CH>(ds, intermesh_egress_index, src_addr, size_bytes, trid);
                record |= E1_SLOT_READ_FWD;
                record_2d_forwarded_slots<CH>(1u << intermesh_egress_index);
            } else {
                uint32_t forwarded_slot_mask;
#if defined(FABRIC_2D_VC1_ACTIVE)
                if constexpr (C::uses_shared_bodies) {
                    if (fwd_slots != 0) {
                        e1_forward_slots_shared<C::data_cmd_buf>(fwd_slots, ds.data(), src_addr, size_bytes, trid);
                        record |= E1_SLOT_READ_FWD;
                    }
                    forwarded_slot_mask = fwd_slots;
                } else
#endif
                {
                    forwarded_slot_mask = e1_forward_2d<CH>(action, src_addr, size_bytes, trid, ds, record);
                }
                record_2d_forwarded_slots<CH>(static_cast<uint8_t>(forwarded_slot_mask));
                if (local_deliver) {
                    e1_local_delivery<CH>(
                        packet_header,
                        hdr_w40.payload_size_bytes,
                        hdr_w40.noc_send_type,
                        trid,
                        C::defers_flushed_atomics ? &st.pending_atomic : nullptr,
                        record);
                }
            }
#else
            constexpr bool stateful_1d =
#if !defined(DEBUG_PRINT_ENABLED) and !defined(WATCHER_ENABLED)
                !FORCE_ALL_PATHS_TO_USE_SAME_NOC;
#else
                false;
#endif
            // 1D: WRITE_ONLY delivers locally, FORWARD_ONLY forwards, WRITE_AND_FORWARD forwards and then delivers
            // (a sparse multicast write delivers first: see below), NOOP does nothing.
            const auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_header);
            const uint16_t payload_size_bytes = packed.payload_size_bytes;
            const tt::tt_fabric::NocSendType noc_send_type = packed.noc_send_type;
            const uint32_t src_addr = reinterpret_cast<uint32_t>(packet_header);
            const uint32_t size_bytes = payload_size_bytes + sizeof(PACKET_HEADER_TYPE);
            const bool deliver_locally = (routing & LowLatencyFields::WRITE_ONLY) != 0;
            const bool forward = (routing & LowLatencyFields::FORWARD_ONLY) != 0;
            ASSERT(routing != LowLatencyFields::NOOP);
            const bool sparse_write_and_forward = deliver_locally && forward &&
                                                  noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE;
            if (sparse_write_and_forward) [[unlikely]] {
                // This chip writes its own page group, then write_idx/chip_idx advance so the downstream chip picks the
                // next group; the forward below sends the advanced header.
                generic_rx_local_delivery_impl(packet_header, payload_size_bytes, noc_send_type, trid, CH);
                auto& sparse = packet_header->command_fields.sparse_mcast_write;
                sparse.write_idx += sparse.counts[sparse.chip_idx];
                sparse.chip_idx++;
            }
            if (forward) {
                update_packet_header_for_next_hop(packet_header, cached_routing_fields);
                e1_forward_to<CH, 0, stateful_1d>(ds[0], src_addr, size_bytes, trid);
                record |= E1_SLOT_READ_FWD;
                channel_trimming_usage_recorder.set_sender_channel_forwarded_to(CH, 1);
            }
            if (deliver_locally) {
                if (sparse_write_and_forward) {
                    record |= E1_SLOT_READ_LOCAL;  // delivered above
                } else {
                    e1_local_delivery<CH>(packet_header, payload_size_bytes, noc_send_type, trid, nullptr, record);
                }
            }
#endif
            records.set(st.idx, record);
            e1_retire<CH>(st);
        }
    }

    if constexpr (C::defers_flushed_atomics) {
        if (!progress && st.pending_atomic) {
            drain_pending_atomic_flush(CH, st.pending_atomic);
        }
    }
    receiver_forward_timer.close();

    e1_rx_completions<CH>(st, records, trid_tracker, credit_sender, progress);
    return progress;
}

// E1's downstream connections, set up exactly as kernel_main's RouterToRouterSender adapters are for a receiver ERISC,
// from the same runtime args (per VC: has-downstream mask with one bit per compact slot; ring bases, NoC x/y,
// registration and location-info addresses and teardown semaphores by dense index), VC0 first, then VC1:
//   e1_init_downstreams (at adapter construction, before the Ethernet handshake): per connected downstream, zero its
//     teardown semaphore and set the local copy of its free slots to its slot count; fill the cursor (slot i of the
//     downstream ring is base + i * channel_buffer_size; stateless sends go to get_noc_addr(x, y, 0) >>
//     NOC_ADDR_COORD_SHIFT, stateless credits to the downstream's free-slots update register).
//   e1_open_downstreams (after the handshake, VCs whose receiver channel is serviced): connection::open_start, then
//     open_finish, per connected downstream.
//   e1_setup_downstream_cmd_bufs (after the next sync, same VCs): pre-program the data and sync command buffers per
//     connected downstream, then VC0's downstream 0 again when the stateful single-live forward uses it.
// Each runs once, out of line, over a small table of the VCs (one body for all of them).
struct E1DownstreamSetup {
    E1DownstreamCursor* cursors;
    uint32_t size;  // compact slots
    uint32_t has_downstream_mask;
    const uint32_t* ring_bases;
    const uint32_t* location_info_addrs;
    const uint32_t* registration_addrs;
    const size_t* teardown_sem_addrs;
    uint32_t noc_x_packed;
    uint32_t noc_y_packed;
    const uint32_t* remote_credit_streams;  // by compact slot
    const uint32_t* local_credit_streams;   // by compact slot
    uint32_t num_slots;
    uint8_t data_cmd_buf;
    uint8_t sync_cmd_buf;
    bool receiver_serviced;
};

template <size_t VC>
FORCE_INLINE E1DownstreamSetup e1_downstream_setup(
    std::array<E1DownstreamCursor, E1Downstreams<VC>::size>& ds,
    uint32_t has_downstream_edm_mask,
    const uint32_t* ring_bases,
    const uint32_t* location_info_addrs,
    const uint32_t* registration_addrs,
    const size_t* teardown_sem_addrs,
    uint32_t noc_x_packed,
    uint32_t noc_y_packed) {
    using D = E1Downstreams<VC>;
    return E1DownstreamSetup{
        ds.data(),
        D::size,
        has_downstream_edm_mask & 0xF,
        ring_bases,
        location_info_addrs,
        registration_addrs,
        teardown_sem_addrs,
        noc_x_packed,
        noc_y_packed,
        D::remote_credit_streams.data(),
        D::local_credit_streams.data(),
        D::num_slots,
        D::data_cmd_buf,
        D::sync_cmd_buf,
        is_receiver_channel_serviced[VC]};
}

template <bool UNUSED = true>
__attribute__((noinline)) void e1_init_downstreams(const E1DownstreamSetup* vcs, uint32_t num_vcs) {
    for (uint32_t v = 0; v < num_vcs; v++) {
        const E1DownstreamSetup& vc = vcs[v];
        uint32_t dense_index = 0;
        for (uint32_t compact_index = 0; compact_index < vc.size; compact_index++) {
            E1DownstreamCursor& c = vc.cursors[compact_index];
            c.free_slots = 0;
            c.local_credit_read_addr = get_stream_reg_read_addr(vc.local_credit_streams[compact_index]);
            c.local_credit_update_addr = get_stream_reg_write_addr(vc.local_credit_streams[compact_index]);
            if ((vc.has_downstream_mask >> compact_index) & 0x1) {
                *reinterpret_cast<volatile uint32_t*>(vc.teardown_sem_addrs[dense_index]) = 0;
                init_ptr_val(vc.local_credit_streams[compact_index], vc.num_slots);
                const uint32_t ring_base = vc.ring_bases[dense_index];
                const uint8_t noc_x = (vc.noc_x_packed >> (dense_index * 8)) & 0xFF;
                const uint8_t noc_y = (vc.noc_y_packed >> (dense_index * 8)) & 0xFF;
                c.slot_addr = ring_base;
                c.first_slot_addr = ring_base;
                c.last_slot_addr = ring_base + (vc.num_slots - 1) * channel_buffer_size;
                c.noc_coord = static_cast<uint32_t>(get_noc_addr(noc_x, noc_y, 0) >> NOC_ADDR_COORD_SHIFT);
                c.credit_noc_addr = get_noc_addr(
                    noc_x,
                    noc_y,
                    get_stream_reg_write_addr(vc.remote_credit_streams[compact_index]),
                    tt::tt_fabric::edm_to_downstream_noc);
                dense_index++;
            } else {
                c.slot_addr = 0;
                c.first_slot_addr = 0;
                c.last_slot_addr = 0;
                c.noc_coord = 0;
                c.credit_noc_addr = 0;
            }
        }
    }
}

template <bool UNUSED = true>
__attribute__((noinline)) void e1_open_downstreams(const E1DownstreamSetup* vcs, uint32_t num_vcs) {
    // Blackhole opens router-to-router connections with non-posted writes (kernel_main's
    // use_posted_writes_for_connection_open).
    constexpr bool posted = false;
    for (uint32_t v = 0; v < num_vcs; v++) {
        const E1DownstreamSetup& vc = vcs[v];
        if (!vc.receiver_serviced) {
            continue;
        }
        uint32_t dense_index = 0;
        for (uint32_t compact_index = 0; compact_index < vc.size; compact_index++) {
            if ((vc.has_downstream_mask >> compact_index) & 0x1) {
                const uint8_t noc_x = (vc.noc_x_packed >> (dense_index * 8)) & 0xFF;
                const uint8_t noc_y = (vc.noc_y_packed >> (dense_index * 8)) & 0xFF;
                tt::tt_fabric::connection::open_start<false, posted, tt::tt_fabric::worker_handshake_noc>(
                    vc.location_info_addrs[dense_index],
                    vc.cursors[compact_index].local_credit_update_addr,
                    vc.teardown_sem_addrs[dense_index],
                    noc_x,
                    noc_y);
                tt::tt_fabric::connection::open_finish<posted, tt::tt_fabric::worker_handshake_noc>(
                    vc.registration_addrs[dense_index],
                    reinterpret_cast<volatile uint32_t*>(vc.teardown_sem_addrs[dense_index]),
                    noc_x,
                    noc_y);
                dense_index++;
            }
        }
    }
}

template <bool UNUSED = true>
__attribute__((noinline)) void e1_setup_downstream_cmd_bufs(
    const E1DownstreamSetup* vcs, uint32_t num_vcs, bool vc0_downstream0_again) {
    constexpr uint8_t noc = tt::tt_fabric::edm_to_downstream_noc;
    constexpr uint8_t noc_vc = tt::tt_fabric::forward_and_local_write_noc_vc;
    auto setup = [](const E1DownstreamSetup& vc, uint32_t compact_index, uint32_t dense_index) {
        const uint8_t noc_x = (vc.noc_x_packed >> (dense_index * 8)) & 0xFF;
        const uint8_t noc_y = (vc.noc_y_packed >> (dense_index * 8)) & 0xFF;
        const uint64_t edm_noc_addr = get_noc_addr(noc_x, noc_y, 0, noc);
        noc_async_write_one_packet_with_trid_set_state<true>(edm_noc_addr, vc.data_cmd_buf, noc, noc_vc);
        const uint64_t noc_sem_addr =
            get_noc_addr(noc_x, noc_y, get_stream_reg_write_addr(vc.remote_credit_streams[compact_index]), noc);
        noc_inline_dw_write_set_state<true, true>(
            noc_sem_addr, pack_value_for_inc_on_write_stream_reg_write(-1), 0xF, vc.sync_cmd_buf, noc, noc_vc);
    };
    for (uint32_t v = 0; v < num_vcs; v++) {
        const E1DownstreamSetup& vc = vcs[v];
        if (!vc.receiver_serviced) {
            continue;
        }
        uint32_t dense_index = 0;
        for (uint32_t compact_index = 0; compact_index < vc.size; compact_index++) {
            if ((vc.has_downstream_mask >> compact_index) & 0x1) {
                setup(vc, compact_index, dense_index);
                dense_index++;
            }
        }
        if (v == 0 && vc0_downstream0_again && (vc.has_downstream_mask & 0x1)) {
            setup(vc, 0, 0);
        }
    }
}

// E1's main loop: every iterations_between_ctx_switch_and_teardown_checks passes, check termination, drain a deferred
// atomic, update telemetry and the heartbeat, and follow the context-switch handshake (E1 is never its master).
template <
    typename TridTrackerCH0
#if defined(FABRIC_2D_VC1_ACTIVE)
    ,
    typename TridTrackerCH1
#endif
    >
FORCE_INLINE void run_e1_receive_program(
    [[maybe_unused]] std::array<E1DownstreamCursor, E1Downstreams<0>::size>& ds_vc0,
#if defined(FABRIC_2D_VC1_ACTIVE)
    [[maybe_unused]] std::array<E1DownstreamCursor, E1Downstreams<1>::size>& ds_vc1,
#endif
    TridTrackerCH0& trid_tracker_ch0,
#if defined(FABRIC_2D_VC1_ACTIVE)
    [[maybe_unused]] TridTrackerCH1& trid_tracker_ch1,
#endif
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr,
    [[maybe_unused]] std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids) {
    FabricTelemetry local_fabric_telemetry{};
    [[maybe_unused]] auto fabric_telemetry = reinterpret_cast<volatile FabricTelemetry*>(MEM_AERISC_FABRIC_TELEMETRY_BASE);

    const auto& routing_table =
        *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const uint8_t my_mesh_coord_y = routing_table.my_mesh_coord_y;
    const uint8_t my_mesh_coord_x = routing_table.my_mesh_coord_x;
    const uint16_t my_mesh_id = routing_table.my_mesh_id;

    auto credit_senders = init_receiver_channel_response_credit_senders<NUM_RECEIVER_CHANNELS>();

#if defined(FABRIC_2D_VC0_CROSSOVER_TO_VC1)
    auto& ds_ch0 = ds_vc1;
#else
    auto& ds_ch0 = ds_vc0;
#endif
#if defined(FABRIC_2D_VC1_SERVICED)
    auto& ds_ch1 = ds_vc1;
#endif
    E1RxState st_ch0;
    e1_rx_state_init<0>(st_ch0);
    E1SlotRecords<E1Rx<0>::num_slots> records_ch0{};
#if defined(FABRIC_2D_VC1_SERVICED)
    E1RxState st_ch1;
    e1_rx_state_init<1>(st_ch1);
    E1SlotRecords<E1Rx<1>::num_slots> records_ch1{};
#endif

    uint16_t fabric_heartbeat_counter = 0;
    constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_ADDR = 0x7CC70;
    volatile uint32_t* fabric_heartbeat_ptr = reinterpret_cast<volatile uint32_t*>(FABRIC_KERNEL_HEARTBEAT_ADDR);

    // STABILITY_PROBE_PRELOOP
    while (!got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
        uint32_t rx_progress = 0;
        uint64_t loop_start_cycles;
        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            loop_start_cycles = get_timestamp();
        }
        for (size_t i = 0; i < iterations_between_ctx_switch_and_teardown_checks; i++) {
            // STABILITY_PROBE_LOOP
            rx_progress |= e1_rx_step<0>(
                st_ch0,
                records_ch0,
                ds_ch0,
                trid_tracker_ch0,
                credit_senders.template get<0>(),
                routing_table,
                my_mesh_id,
                my_mesh_coord_y,
                my_mesh_coord_x,
                local_fabric_telemetry);
#if defined(FABRIC_2D_VC1_SERVICED)
            if constexpr (is_receiver_channel_serviced[1]) {
                rx_progress |= e1_rx_step<1>(
                    st_ch1,
                    records_ch1,
                    ds_ch1,
                    trid_tracker_ch1,
                    credit_senders.template get<1>(),
                    routing_table,
                    my_mesh_id,
                    my_mesh_coord_y,
                    my_mesh_coord_x,
                    local_fabric_telemetry);
            }
#endif
        }
        if constexpr (E1Rx<0>::defers_flushed_atomics) {
            if (st_ch0.pending_atomic) {
                drain_pending_atomic_flush(0, st_ch0.pending_atomic);
            }
        }

        if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
            uint64_t loop_end_cycles = get_timestamp();
            uint64_t loop_delta_cycles = loop_end_cycles - loop_start_cycles;
            update_bw_cycles(loop_delta_cycles, /*tx_progress=*/false, rx_progress, local_fabric_telemetry);
        }
        if constexpr (FABRIC_TELEMETRY_ANY_DYNAMIC_STAT) {
            update_telemetry(
                local_sender_channel_free_slots_stream_ids,
                /*tx_progress=*/false,
                rx_progress,
                local_fabric_telemetry,
                fabric_telemetry);
        }

        if ((++fabric_heartbeat_counter & 0x3F) == 0) {
            *fabric_heartbeat_ptr = 0xDCBA0000 | fabric_heartbeat_counter;
        }

        run_routing_without_noc_sync_coordinated_as_non_master(termination_signal_ptr);
    }
}
#endif  // ARCH_BLACKHOLE

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

// ============================================================================
// The sending ERISC's program (ERISC 0 in two-ERISC mode)
// ============================================================================
// In two-ERISC mode ERISC 0 services only sender channels. It runs `e0::run_main_loop`, its own function with its own
// state, in place of the generic loop above. One pass:
//   1. Read VC0's free-slots registers as one fenced group (channels 0-3, as the generic loop does).
//   2. Worker channel (channel 0), specialized at compile time: send if it has a packet and the far ring has room,
//      take completions, return credits to the worker, poll the connection semaphore (open or tear down the
//      connection out of line when it changes).
//   3. VC0's forwarding channels: one test says whether any has work (an unsent packet, or packets in flight). Only
//      then is each forwarding channel tested on its own, in channel order, and a channel with work visited by code
//      specialized for it at compile time and inlined here. "All slots free" is the generic loop's test for that
//      channel (all_slots_free), bit test or masked compare, in the combined test and in each channel's test alike.
//   4. VC1 (when serviced): its free-slots registers as a second group, then the same test and visits.
// The pass makes no calls while it sends, takes completions and returns credits (as in the generic loop). Out of line
// and off the pass's path: worker connection open and close, and the static-connection wait at init.
//
// State held across passes: per VC, the far ring's free count and slot pointer; for the worker, its slot, sent and
// completed counts, the connection semaphore pointer and the value it holds while nothing changes; one word with a
// bit per forwarding channel that has packets in flight. A forwarding channel's counts, slot and upstream credit
// address are a record its visit reads and writes. Addresses and slot bounds known at compile time are constants in
// the code, not state.
//
// Init: this ERISC sets up its own (sender) side. The receiver side (downstream router adapters, receiver channel
// buffers) is left to the receiving ERISC, which has the same compile-time and runtime args and performs the same
// writes in the same init phase.
//
// Load groups (reads issued back to back so their latencies overlap) use the generic loop's fences: the grouped
// free-slots read, and send_packet's pre-send and post-send groups, which are sender_channel_try_send's.
//
// Same protocol as the generic loop: every packet, register update, credit, completion and connection step, in the
// same order on each channel, channels visited in the same order within a pass. Builds this program does not cover
// keep the generic loop: single-ERISC mode, ERISC 1, the speedy path, VC2, stream-register credits, the RISC data
// cache, spin-wait sends, non-Blackhole targets, a fifth VC0 sender, a forwarding-type channel 0 and a worker-type
// forwarding channel.
namespace e0 {

#if defined(FABRIC_2D_VC1_SERVICED)
constexpr bool vc1_serviced = is_2d_fabric != 0;
#else
constexpr bool vc1_serviced = false;
#endif
#if defined(FABRIC_2D_VC2_SERVICED)
constexpr bool vc2_serviced = true;
#else
constexpr bool vc2_serviced = false;
#endif

// Channels this program visits: VC0's first two (1D) or four (2D), and VC1's when VC1 is serviced; serviced ones only.
constexpr size_t vc0_visit_limit = is_2d_fabric ? 4 : 2;
constexpr size_t vc0_end =
    ACTUAL_VC0_SENDER_CHANNELS < vc0_visit_limit ? ACTUAL_VC0_SENDER_CHANNELS : vc0_visit_limit;
constexpr size_t vc1_begin = VC1_LOCAL_CHANNEL_START;
constexpr size_t vc1_end =
    vc1_serviced ? VC1_LOCAL_CHANNEL_START + ACTUAL_VC1_SENDER_CHANNELS : VC1_LOCAL_CHANNEL_START;

constexpr bool visits_on_vc0(size_t ch) {
    return ch < NUM_SENDER_CHANNELS && is_sender_channel_serviced[ch] && ch < vc0_end;
}
constexpr bool visits_on_vc1(size_t ch) {
    return ch < NUM_SENDER_CHANNELS && is_sender_channel_serviced[ch] && ch >= vc1_begin && ch < vc1_end;
}
constexpr uint32_t forwarding_channels_on_vc0() {
    uint32_t mask = 0;
    for (size_t ch = 1; ch < MAX_NUM_SENDER_CHANNELS; ch++) {
        mask |= visits_on_vc0(ch) ? (1u << ch) : 0u;
    }
    return mask;
}
constexpr uint32_t forwarding_channels_on_vc1() {
    uint32_t mask = 0;
    for (size_t ch = 1; ch < MAX_NUM_SENDER_CHANNELS; ch++) {
        mask |= visits_on_vc1(ch) ? (1u << ch) : 0u;
    }
    return mask;
}
constexpr bool has_worker = visits_on_vc0(0);
constexpr uint32_t vc0_forwarding = forwarding_channels_on_vc0();
constexpr uint32_t vc1_forwarding = forwarding_channels_on_vc1();
// VC1's far ring (receiver channel); VC0's when VC1 has no channels here, so that the type always exists.
constexpr size_t vc1_far_ring = vc1_forwarding != 0 ? VC1_RECEIVER_CHANNEL : VC0_RECEIVER_CHANNEL;

// The generic loop's visits (run_fabric_edm_main_loop), to check that this program visits the same channels.
constexpr bool generic_loop_visits_on_vc0(size_t ch) {
    return ch == 0 || ch == 1 || (is_2d_fabric && (ch == 2 || ch == 3 || (ch == 4 && ACTUAL_VC0_SENDER_CHANNELS > 4)));
}
constexpr bool generic_loop_visits_on_vc1(size_t ch) {
    return vc1_serviced && ch >= ACTUAL_VC0_SENDER_CHANNELS &&
           ch < ACTUAL_VC0_SENDER_CHANNELS + (ACTUAL_VC1_SENDER_CHANNELS > 3 ? 4 : 3);
}

constexpr bool shape_supported = [] {
    for (size_t ch = 0; ch < MAX_NUM_SENDER_CHANNELS; ch++) {
        const bool serviced = is_sender_channel_serviced[ch];
        if ((serviced && generic_loop_visits_on_vc0(ch)) != visits_on_vc0(ch) ||
            (serviced && generic_loop_visits_on_vc1(ch)) != visits_on_vc1(ch)) {
            return false;
        }
        if (visits_on_vc0(ch) || visits_on_vc1(ch)) {
            // Channel 0 is the worker channel (connection-checked); every other visited channel is router-to-router.
            if (sender_ch_live_check_skip[ch] != (ch != 0) || !sender_channel_uses_counter_credits(ch) ||
                SENDER_NUM_BUFFERS_ARRAY[ch] == 0 || SENDER_NUM_BUFFERS_ARRAY[ch] > grouped_free_slots_imm_mask) {
                return false;
            }
        }
    }
    return true;
}();

// Whether this ERISC runs this program.
constexpr bool runs = MY_ERISC_ID == 0 && NUM_ACTIVE_ERISCS > 1 && !any_receiver_channel_serviced &&
                      any_sender_channel_serviced && !super_speedy_mode && is_blackhole_arch && multi_txq_enabled &&
                      !ENABLE_RISC_CPU_DATA_CACHE && !ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA && !vc2_serviced &&
                      shape_supported;

// Injection channels under bubble flow control send only with two free far slots.
constexpr uint32_t bubble_flow_control_channels() {
    uint32_t mask = 0;
    for (size_t ch = 0; ch < NUM_SENDER_CHANNELS; ch++) {
        mask |= (enable_deadlock_avoidance && sender_channel_is_traffic_injection_channel[ch]) ? (1u << ch) : 0u;
    }
    return mask;
}
constexpr uint32_t bubble_channels = bubble_flow_control_channels();

// "Every slot of channel CH is free", exactly as the generic loop decides it, so that every value of the free-slots
// register gives the same answer as there (not only values in [0, num_slots]):
//   - forwarding channels in the grouped read that the generic loop idle-skips: channel_all_slots_free (one bit test
//     when skip_idle_bit_test, else the masked compare);
//   - other channels in the grouped read (the worker channel, forwarding channels without idle skip): the count masked
//     with grouped_free_slots_imm_mask, compared;
//   - channels outside the grouped read (VC1): the full free-slots field (get_ptr_val), compared.
constexpr bool idle_skipped(size_t ch) {
    return ch < grouped_packet_check_channels && skip_idle_per_channel && skip_idle_channel(ch);
}
constexpr bool uses_bit_test(size_t ch) { return idle_skipped(ch) && skip_idle_bit_test; }
constexpr uint32_t free_slots_compare_mask(size_t ch) {
    return ch < grouped_packet_check_channels ? grouped_free_slots_imm_mask : (1u << REMOTE_DEST_WORDS_FREE_WIDTH) - 1;
}
template <size_t CH>
FORCE_INLINE bool all_slots_free(uint32_t raw) {
    constexpr uint32_t n = SENDER_NUM_BUFFERS_ARRAY[CH];
    static_assert(n <= grouped_free_slots_imm_mask);
    if constexpr (idle_skipped(CH)) {
        return channel_all_slots_free<CH>(raw);
    } else {
        return (raw & free_slots_compare_mask(CH)) == n;
    }
}
// The generic loop tests every channel of `mask` the same way (a VC's forwarding channels are all in the grouped read,
// VC0's, or all outside it, VC1's; idle skip is decided for all grouped forwarding channels at once).
constexpr bool tested_alike(uint32_t mask) {
    const size_t first = __builtin_ctz(mask);
    for (size_t ch = first; ch < NUM_SENDER_CHANNELS; ch++) {
        if (((mask >> ch) & 1) && (uses_bit_test(ch) != uses_bit_test(first) ||
                                   free_slots_compare_mask(ch) != free_slots_compare_mask(first))) {
            return false;
        }
    }
    return true;
}

template <size_t... K, typename F>
FORCE_INLINE void for_each_index(std::index_sequence<K...>, F&& f) {
    (f(std::integral_constant<size_t, K>{}), ...);
}

// Channel CH's slots in its L1 ring (the values the sender channel objects are initialized with).
template <size_t CH>
struct ChannelSlots {
    static constexpr uint32_t num_slots = SENDER_NUM_BUFFERS_ARRAY[CH];
    static constexpr uint32_t first = channel_allocs::Entry<SENDER_TO_ENTRY_IDX[CH]>::base_address;
    static constexpr uint32_t last = first + (num_slots - 1) * channel_buffer_size;
};

// Counters the far receiver writes over Ethernet, one word per sender channel.
FORCE_INLINE constexpr uint32_t completion_counter_addr(uint32_t ch) {
    return to_sender_remote_completion_counters_base_address + ch * sizeof(uint32_t);
}
FORCE_INLINE constexpr uint32_t ack_counter_addr(uint32_t ch) {
    return to_sender_remote_ack_counters_base_address + ch * sizeof(uint32_t);
}

// A VC's free-slots stream registers are contiguous, so a channel's update register is the VC's first plus an offset.
template <size_t VC_BEGIN>
constexpr uint32_t first_free_slots_stream =
    sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(VC_BEGIN)];
template <size_t VC_BEGIN>
FORCE_INLINE constexpr uint32_t free_slots_update_addr(uint32_t ch) {
    constexpr uint32_t first_stream = first_free_slots_stream<VC_BEGIN>;
    return STREAM_REG_ADDR(first_stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX) +
           (ch - VC_BEGIN) * NOC_STREAM_REG_SPACE_SIZE;
}
template <size_t VC_BEGIN>
constexpr bool free_slots_streams_contiguous(uint32_t channels) {
    for (size_t ch = 0; ch < NUM_SENDER_CHANNELS; ch++) {
        if (((channels >> ch) & 1) == 0) {
            continue;
        }
        const uint32_t stream = sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(ch)];
        if (ch < VC_BEGIN || stream != first_free_slots_stream<VC_BEGIN> + (ch - VC_BEGIN)) {
            return false;
        }
    }
    return true;
}

// The far receiver ring of one VC, shared by all of that VC's sender channels.
template <size_t RX>
struct FarRing {
    static constexpr uint32_t num_slots = REMOTE_RECEIVER_NUM_BUFFERS_ARRAY[RX];
    static constexpr uint32_t first = eth_remote_channel_allocs::Entry<REMOTE_RECEIVER_TO_ENTRY_IDX[RX]>::base_address;
    static constexpr uint32_t last = first + (num_slots - 1) * channel_buffer_size;
    static constexpr uint32_t packets_sent_update_addr =
        STREAM_REG_ADDR(to_receiver_packets_sent_streams[RX], STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);

    uint32_t free_slots = num_slots;
    uint32_t next_slot = first;
};

// A forwarding (router-to-router) channel's state. Read and written only by a visit that has work; its slot bounds and
// register addresses are compile-time constants of that visit.
struct ForwardingChannel {
    uint32_t slot;       // next slot to send from
    uint32_t sent;       // packets sent
    uint32_t completed;  // completions processed
    uint32_t acked;      // first-level acks processed (first-level-ack VCs only)
    uint64_t upstream_credit_addr;  // the upstream router's free-slots register, cached at init
    // STABILITY_PROBE_STATE
};

// Per-send telemetry (bandwidth and perf telemetry), when the build has any.
constexpr bool telemetry_enabled =
    FABRIC_TELEMETRY_ANY_DYNAMIC_STAT || perf_telemetry_mode != PerfTelemetryRecorderType::NONE;
struct SendTelemetry {
    PerfTelemetryRecorder* perf;
    FabricTelemetry* local;
    bool* tx_progress;
};
struct NoSendTelemetry {};
using Telemetry = std::conditional_t<telemetry_enabled, SendTelemetry, NoSendTelemetry>;
template <typename T = Telemetry>
FORCE_INLINE T make_telemetry(PerfTelemetryRecorder& perf, FabricTelemetry& local, bool& tx_progress) {
    if constexpr (std::is_same_v<T, SendTelemetry>) {
        return T{&perf, &local, &tx_progress};
    } else {
        return T{};
    }
}

// What the main loop shares with the forwarding visits.
struct ForwardingState {
    ForwardingChannel channels[NUM_SENDER_CHANNELS];  // indexed by channel; entry 0 (the worker channel) is unused
    uint32_t in_flight;                               // bit per forwarding channel with packets not yet completed
    bool attempted;                                   // a send attempt yielded this round (the TX queue was busy)
    Telemetry telemetry;
};

// The TX queue was busy after a data command (never seen in practice): spin until it is idle.
template <int = 0>
FORCE_INLINE void wait_for_sender_txq_idle() {
    while (internal_::eth_txq_is_busy(sender_txq_id)) {
    }
}

// Sends channel `ch`'s next packet. Precondition: the channel has an unsent packet and the far ring has room. The same
// steps in the same order as sender_channel_try_send's overlapped form: read the payload size and the TX-queue status
// as one group and stamp src_ch_id; a busy queue yields (returns false, nothing sent). Otherwise issue the data
// command, advance the far slot and the channel's slot, then read the queue status and the channel's completion
// counter as one group while the free count and the channel's free-slots register are updated, and increment the far
// receiver's packets-received register on the same queue. `completions` gets the completions newly reported. A send
// attempt that yields sets `attempted`; one that sends counts in `sent` (the idle context switch test reads both).
template <size_t RX, typename TelemetryT>
FORCE_INLINE bool send_packet(
    uint32_t ch,
    uint32_t& slot,
    uint32_t first_slot,
    uint32_t last_slot,
    FarRing<RX>& far,
    uint32_t free_slots_update,
    uint32_t completion_counter,
    uint32_t& sent,
    uint32_t completed,
    int32_t& completions,
    bool& attempted,
    TelemetryT& telemetry) {
    const uint32_t src_addr = slot;
    auto* pkt_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(src_addr);
    constexpr uint32_t txq_base = ETH_TXQ0_REGS_START + sender_txq_id * ETH_TXQ_REGS_SIZE;
    volatile uint32_t* const txq_regs = reinterpret_cast<volatile uint32_t*>(txq_base);

    std::atomic_signal_fence(std::memory_order_seq_cst);
    const uint32_t raw_payload_size = pkt_header->get_payload_size_excluding_header();
    (void)txq_regs[ETH_TXQ_CMD >> 2];
    const uint32_t pre_txq_status = txq_regs[ETH_TXQ_STATUS >> 2];
    std::atomic_signal_fence(std::memory_order_seq_cst);

    if constexpr (!skip_src_ch_id_update) {
        pkt_header->src_ch_id = static_cast<uint8_t>(ch);
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);

    if (static_cast<int32_t>(pre_txq_status << (31 - ETH_TXQ_STATUS_CMD_ONGOING_BIT)) < 0) {
        attempted = true;
        return false;
    }
    const uint32_t payload_size_bytes = raw_payload_size + sizeof(PACKET_HEADER_TYPE);
    record_sender_channel_usage(ch, payload_size_bytes);

    internal_::eth_send_packet_bytes_unsafe(sender_txq_id, src_addr, far.next_slot, payload_size_bytes);
    far.next_slot = far.next_slot == FarRing<RX>::last ? FarRing<RX>::first : far.next_slot + channel_buffer_size;
    slot = src_addr == last_slot ? first_slot : src_addr + channel_buffer_size;
    if constexpr (telemetry_enabled) {
        record_packet_send(*telemetry.perf, ch, payload_size_bytes);
    }

    std::atomic_signal_fence(std::memory_order_seq_cst);
    (void)txq_regs[ETH_TXQ_CMD >> 2];
    const uint32_t txq_status = txq_regs[ETH_TXQ_STATUS >> 2];
    const uint32_t completions_raw = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(completion_counter);
    std::atomic_signal_fence(std::memory_order_seq_cst);

    far.free_slots--;
    *reinterpret_cast<volatile uint32_t*>(free_slots_update) = 1u << REMOTE_DEST_BUF_WORDS_FREE_INC;
    std::atomic_signal_fence(std::memory_order_seq_cst);

    if (static_cast<int32_t>(txq_status << (31 - ETH_TXQ_STATUS_CMD_ONGOING_BIT)) < 0) {
        wait_for_sender_txq_idle();
    }
    txq_regs[ETH_TXQ_DEST_ADDR >> 2] = FarRing<RX>::packets_sent_update_addr;
    txq_regs[ETH_TXQ_REMOTE_REG_DATA >> 2] = 1u << REMOTE_DEST_BUF_WORDS_FREE_INC;
    txq_regs[ETH_TXQ_CMD >> 2] = ETH_TXQ_CMD_START_REG;
    sent++;
    if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
        update_bw_counters(pkt_header, *telemetry.local);
    }
    if constexpr (FABRIC_TELEMETRY_ANY_DYNAMIC_STAT) {
        *telemetry.tx_progress = true;
    }
    completions = static_cast<int32_t>(completions_raw - completed);
    return true;
}

// The far ring has room for channel `ch` of CHANNELS: one slot, or two for an injection channel under bubble flow
// control.
template <size_t RX, uint32_t CHANNELS>
FORCE_INLINE bool far_has_room(const FarRing<RX>& far, uint32_t ch) {
    if constexpr ((bubble_channels & CHANNELS) == 0) {
        return far.free_slots != 0;
    } else if constexpr ((bubble_channels & CHANNELS) == CHANNELS) {
        return far.free_slots >= BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS;
    } else {
        return far.free_slots >=
               (((bubble_channels >> ch) & 1) ? BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS : 1);
    }
}

// Credit back to the upstream router: +n on its free-slots register.
template <typename ChannelT>
FORCE_INLINE void credit_upstream_router(const ChannelT& channel, int32_t n) {
    noc_inline_dw_write<InlineWriteDst::REG, true>(
        channel.upstream_credit_addr,
        pack_value_for_inc_on_write_stream_reg_write(n),
        0xf,
        tt::tt_fabric::worker_handshake_noc);
}

// Credit back to the worker: the absolute count of slots returned, written into the worker's L1 while it is connected
// and into the connection info for the next worker otherwise.
template <typename WorkerInterfaceT>
FORCE_INLINE void credit_worker(const WorkerInterfaceT& worker, uint32_t credits_returned, bool connected) {
    if (connected) {
        noc_inline_dw_write<InlineWriteDst::L1, true, enable_read_counter_update_noc_flush>(
            worker.cached_worker_semaphore_address,
            credits_returned,
            0xf,
            tt::tt_fabric::worker_handshake_noc,
            NOC_UNICAST_WRITE_VC,
            notify_worker_of_read_counter_update_src_address);
    } else {
        reinterpret_cast<volatile tt_l1_ptr EDMChannelWorkerLocationInfo*>(local_sender_channel_0_connection_info_addr)
            ->edm_read_counter = credits_returned;
    }
}

// The worker connection semaphore holds `expected` while nothing changes: unused_connection_value with no worker
// connected, open_connection_value with one. Any other value comes here: open on a request (open, or open then close
// already), tear down on a close request. Returns the new expected value. The worker interface's read counter is set
// to the credits returned so far, the value it holds in the generic loop.
template <typename WorkerInterfaceT>
__attribute__((noinline)) uint32_t worker_connection_event(
    WorkerInterfaceT& worker, uint32_t semaphore, uint32_t expected, uint32_t credits_returned) {
    worker.local_read_counter.counter = credits_returned;
    if (expected == tt::tt_fabric::connection_interface::unused_connection_value) {
        if (connect_is_requested(semaphore)) {
            establish_worker_connection<MY_ETH_CHANNEL, ENABLE_RISC_CPU_DATA_CACHE>(worker);
            return tt::tt_fabric::connection_interface::open_connection_value;
        }
        return tt::tt_fabric::connection_interface::unused_connection_value;
    }
    if (semaphore == tt::tt_fabric::connection_interface::close_connection_request_value) {
        worker.template teardown_worker_connection<true, ENABLE_RISC_CPU_DATA_CACHE>();
        return tt::tt_fabric::connection_interface::unused_connection_value;
    }
    return tt::tt_fabric::connection_interface::open_connection_value;
}
static_assert(
    tt::tt_fabric::connection_interface::open_connection_value !=
    tt::tt_fabric::connection_interface::unused_connection_value);

// One visit to forwarding channel CH, which has work (the generic loop's run_forwarding_sender_step): send if it has a
// packet and the far ring has room, take completions (and first-level acks), credit the upstream router, and keep the
// channel's bit in `in_flight` equal to "packets not yet completed (or acked)". Inline and specialized for CH: slot
// bounds, register and counter addresses and src_ch_id are constants; the channel's counts and slot come from its
// record, read only here.
template <size_t CH, size_t RX, size_t VC_BEGIN, bool FIRST_LEVEL_ACK>
FORCE_INLINE void visit_forwarding_channel(bool has_unsent_packet, FarRing<RX>& far, ForwardingState& state) {
    constexpr uint32_t bit = 1u << CH;
    ForwardingChannel& channel = state.channels[CH];
    uint32_t sent = channel.sent;
    uint32_t completed = channel.completed;
    int32_t completions = 0;
    bool sent_packet = false;
    if (far_has_room<RX, bit>(far, CH) && has_unsent_packet) {
        uint32_t slot = channel.slot;
        sent_packet = send_packet<RX>(
            CH,
            slot,
            ChannelSlots<CH>::first,
            ChannelSlots<CH>::last,
            far,
            free_slots_update_addr<VC_BEGIN>(CH),
            completion_counter_addr(CH),
            sent,
            completed,
            completions,
            state.attempted,
            state.telemetry);
        if (sent_packet) {
            channel.slot = slot;
            channel.sent = sent;
        }
    }
    if (!sent_packet && sent != completed) {
        completions = static_cast<int32_t>(
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(completion_counter_addr(CH)) - completed);
    }
    if (completions) {
        far.free_slots += completions;
        completed += completions;
        channel.completed = completed;
        if constexpr (!FIRST_LEVEL_ACK) {
            credit_upstream_router(channel, completions);
        }
    }
    bool busy = sent != completed;
    if constexpr (FIRST_LEVEL_ACK) {
        uint32_t acked = channel.acked;
        if (sent != acked) {
            const uint32_t acks = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_counter_addr(CH)) - acked;
            if (acks > 0) {
                acked += acks;
                channel.acked = acked;
                credit_upstream_router(channel, acks);
            }
        }
        busy = busy || sent != acked;
    }
    state.in_flight = busy ? (state.in_flight | bit) : (state.in_flight & ~bit);
}

// The forwarding channels of one VC (bits of CHANNELS), whose free-slots values are raw[k] for channel BASE + k: each
// channel with work is visited, in channel order. With COMBINED_TEST, one test first decides whether any has work.
template <size_t RX, size_t VC_BEGIN, uint32_t CHANNELS, size_t BASE, bool FIRST_LEVEL_ACK, bool COMBINED_TEST>
FORCE_INLINE void visit_forwarding_channels(const uint32_t (&raw)[4], FarRing<RX>& far, ForwardingState& state) {
    static_assert(
        (CHANNELS >> BASE) < 16 && (CHANNELS & ((1u << BASE) - 1)) == 0,
        "a VC's forwarding channels must be in its group of four reads");
    static_assert(free_slots_streams_contiguous<VC_BEGIN>(CHANNELS));
    static_assert(
        (bubble_channels & CHANNELS) == 0 || FIRST_LEVEL_ACK,
        "Bubble flow control and first level ack must be set to the same values");

    if constexpr (COMBINED_TEST) {
        // The one test: every forwarding channel has all its slots free (all_slots_free, the generic loop's test),
        // and none has packets in flight.
        static_assert(tested_alike(CHANNELS));
        constexpr size_t FIRST = __builtin_ctz(CHANNELS);
        bool all_free;
        if constexpr (uses_bit_test(FIRST)) {
            // One shared power-of-two slot count (skip_idle_bit_test requires it): its bit is set in the AND of the
            // values exactly when it is set in each.
            uint32_t all = SENDER_NUM_BUFFERS_ARRAY[FIRST];
            for_each_index(std::make_index_sequence<4>{}, [&](auto k) __attribute__((always_inline)) {
                constexpr size_t K = decltype(k)::value;
                if constexpr (((CHANNELS >> (BASE + K)) & 1) != 0) {
                    all &= raw[K];
                }
            });
            all_free = all != 0;
        } else {
            // (raw & mask) == slots for every channel: the OR of (raw ^ slots), masked, is zero.
            uint32_t differ = 0;
            for_each_index(std::make_index_sequence<4>{}, [&](auto k) __attribute__((always_inline)) {
                constexpr size_t K = decltype(k)::value;
                if constexpr (((CHANNELS >> (BASE + K)) & 1) != 0) {
                    differ |= raw[K] ^ SENDER_NUM_BUFFERS_ARRAY[BASE + K];
                }
            });
            all_free = (differ & free_slots_compare_mask(FIRST)) == 0;
        }
        if (all_free && (state.in_flight & CHANNELS) == 0) {
            return;
        }
    }

    // Each forwarding channel with work (an unsent packet, or packets in flight), in channel order.
    for_each_index(std::make_index_sequence<4>{}, [&](auto k) __attribute__((always_inline)) {
        constexpr size_t K = decltype(k)::value;
        constexpr size_t CH = BASE + K;
        if constexpr (((CHANNELS >> CH) & 1) != 0) {
            const bool has_unsent_packet = !all_slots_free<CH>(raw[K]);
            if (has_unsent_packet || (state.in_flight & (1u << CH)) != 0) {
                visit_forwarding_channel<CH, RX, VC_BEGIN, FIRST_LEVEL_ACK>(has_unsent_packet, far, state);
            }
        }
    });
}

// The VC1 channels' free-slots registers, read back to back after VC0's work (four at most, so the reads overlap).
template <uint32_t CHANNELS, size_t BASE>
FORCE_INLINE void read_free_slots_group(uint32_t (&raw)[4]) {
    for_each_index(std::make_index_sequence<4>{}, [&](auto k) __attribute__((always_inline)) {
        constexpr size_t K = decltype(k)::value;
        if constexpr (((CHANNELS >> (BASE + K)) & 1) != 0) {
            raw[K] = read_free_slots_raw<BASE + K>();
        }
    });
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

// Wait for each router-to-router channel's upstream router to connect, then cache its credit address (the generic
// path's wait_for_static_connection_to_ready, with one poll loop shared by all channels).
template <int = 0>
__attribute__((noinline)) void wait_for_connect_request(
    volatile tt_l1_ptr uint32_t* connection_live_semaphore,
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    uint32_t count = 0;
    while (!connect_is_requested(*connection_live_semaphore) &&
           !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        if (count++ == CONTEXT_SWITCH_TIMEOUT_SHORT) {
            count = 0;
            run_routing();
        }
    }
}

template <typename EdmChannelWorkerIFs>
FORCE_INLINE void wait_for_static_connections(
    EdmChannelWorkerIFs& local_sender_channel_worker_interfaces,
    volatile tt::tt_fabric::TerminationSignal* termination_signal_ptr) {
    tuple_for_each_constexpr(
        local_sender_channel_worker_interfaces.channel_worker_interfaces, [&](auto& interface, auto idx) {
            if constexpr (is_sender_channel_serviced[idx] && sender_ch_wait_static_connection[idx]) {
                wait_for_connect_request(interface.connection_live_semaphore, termination_signal_ptr);
                interface.template cache_producer_noc_addr<ENABLE_RISC_CPU_DATA_CACHE, USE_DYNAMIC_CREDIT_ADDR>();
            }
        });
}

// The sending ERISC's main loop: router commands, the outer loop (termination, telemetry, heartbeat, idle context
// switch) and the inner loop of passes over the sender channels.
template <typename EdmChannelWorkerIFs>
__attribute__((noinline)) void run_main_loop(
    EdmChannelWorkerIFs& local_sender_channel_worker_interfaces,
    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids) {
    auto termination_signal_ptr = reinterpret_cast<volatile tt::tt_fabric::TerminationSignal*>(termination_signal_addr);
    const auto& routing_table =
        *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    auto* state_manager_l1 = const_cast<tt_l1_ptr RouterStateManager*>(&routing_table.state_manager);

    FabricTelemetry local_fabric_telemetry{};
    auto fabric_telemetry = reinterpret_cast<volatile FabricTelemetry*>(MEM_AERISC_FABRIC_TELEMETRY_BASE);
    PerfTelemetryRecorder perf_telemetry_recorder = build_perf_telemetry_recorder<perf_telemetry_mode>();
    auto local_perf_telemetry_buffer =
        build_perf_telemetry_buffer(reinterpret_cast<uint32_t*>(perf_telemetry_buffer_addr));
    bool tx_progress = false;

    FarRing<VC0_RECEIVER_CHANNEL> far_vc0;
    [[maybe_unused]] FarRing<vc1_far_ring> far_vc1;

    // The worker channel. The interface copy serves connection events and holds the worker's credit address.
    constexpr bool worker_first_level_ack = ENABLE_FIRST_LEVEL_ACK_VC0;
    auto worker = local_sender_channel_worker_interfaces.template get<0>();
    volatile tt_l1_ptr uint32_t* const worker_semaphore = worker.connection_live_semaphore;
    uint32_t worker_slot = ChannelSlots<0>::first;
    uint32_t worker_sent = 0;
    uint32_t worker_completed = 0;
    uint32_t worker_acked = 0;
    uint32_t worker_expected = tt::tt_fabric::connection_interface::unused_connection_value;

    // The forwarding channels.
    ForwardingState state{};
    state.telemetry = make_telemetry(perf_telemetry_recorder, local_fabric_telemetry, tx_progress);
    tuple_for_each_constexpr(
        local_sender_channel_worker_interfaces.channel_worker_interfaces, [&](auto& interface, auto idx) {
            constexpr size_t CH = decltype(idx)::value;
            if constexpr ((((vc0_forwarding | vc1_forwarding) >> CH) & 1) != 0) {
                ForwardingChannel& channel = state.channels[CH];
                channel.slot = ChannelSlots<CH>::first;
                channel.upstream_credit_addr = interface.cached_worker_semaphore_address;
            }
        });

    size_t did_nothing_count = 0;
    uint16_t fabric_heartbeat_counter = 0;
    constexpr uint32_t FABRIC_KERNEL_HEARTBEAT_ADDR = 0x7CC70;
    volatile uint32_t* fabric_heartbeat_ptr = reinterpret_cast<volatile uint32_t*>(FABRIC_KERNEL_HEARTBEAT_ADDR);

    // Packets sent on every channel so far. A round attempted a send if this changed or an attempt yielded.
    auto packets_sent = [&]() __attribute__((always_inline)) {
        uint32_t total = worker_sent;
        for_each_index(std::make_index_sequence<NUM_SENDER_CHANNELS>{}, [&](auto c) __attribute__((always_inline)) {
            constexpr size_t CH = decltype(c)::value;
            if constexpr ((((vc0_forwarding | vc1_forwarding) >> CH) & 1) != 0) {
                total += state.channels[CH].sent;
            }
        });
        return total;
    };

    auto execute_main_loop = [&]() __attribute__((always_inline)) {
        // STABILITY_PROBE_PRELOOP
        while (continue_running_main_run_loop<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr, state_manager_l1)) {
            state.attempted = false;
            const uint32_t round_packets_sent = packets_sent();
            tx_progress = false;
            open_perf_recording_window(perf_telemetry_recorder);
            uint64_t loop_start_cycles;
            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                loop_start_cycles = get_timestamp();
            }

            for (size_t i = 0; i < iterations_between_ctx_switch_and_teardown_checks; i++) {
                // STABILITY_PROBE_LOOP
                uint32_t free_slots_raw[4] = {0, 0, 0, 0};
                read_grouped_free_slots(free_slots_raw);

                if constexpr (has_worker) {
                    constexpr uint32_t num_slots = ChannelSlots<0>::num_slots;
                    const bool has_unsent_packet = (free_slots_raw[0] & grouped_free_slots_imm_mask) != num_slots;
                    int32_t completions = 0;
                    bool sent_packet = false;
                    if (far_has_room<VC0_RECEIVER_CHANNEL, 1u>(far_vc0, 0) && has_unsent_packet) {
                        sent_packet = send_packet<VC0_RECEIVER_CHANNEL>(
                            0,
                            worker_slot,
                            ChannelSlots<0>::first,
                            ChannelSlots<0>::last,
                            far_vc0,
                            free_slots_update_addr<0>(0),
                            completion_counter_addr(0),
                            worker_sent,
                            worker_completed,
                            completions,
                            state.attempted,
                            state.telemetry);
                    }
                    if (!sent_packet && worker_sent != worker_completed) {
                        completions = static_cast<int32_t>(
                            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(completion_counter_addr(0)) -
                            worker_completed);
                    }
                    const bool connected =
                        worker_expected != tt::tt_fabric::connection_interface::unused_connection_value;
                    if (completions) {
                        far_vc0.free_slots += completions;
                        worker_completed += completions;
                        if constexpr (!worker_first_level_ack) {
                            credit_worker(worker, worker_completed, connected);
                        }
                    }
                    if constexpr (worker_first_level_ack) {
                        if (worker_sent != worker_acked) {
                            const uint32_t acks =
                                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_counter_addr(0)) - worker_acked;
                            if (acks > 0) {
                                worker_acked += acks;
                                credit_worker(worker, worker_acked, connected);
                            }
                        }
                    }
                    // Worker connection open and teardown, polled on every pass.
                    const uint32_t semaphore = *worker_semaphore;
                    if (semaphore != worker_expected) {
                        worker_expected = worker_connection_event(
                            worker,
                            semaphore,
                            worker_expected,
                            worker_first_level_ack ? worker_acked : worker_completed);
                    }
                }

                if constexpr (vc0_forwarding != 0) {
                    visit_forwarding_channels<
                        VC0_RECEIVER_CHANNEL,
                        0,
                        vc0_forwarding,
                        0,
                        ENABLE_FIRST_LEVEL_ACK_VC0,
                        /*COMBINED_TEST=*/true>(free_slots_raw, far_vc0, state);
                }
                if constexpr (vc1_forwarding != 0) {
                    uint32_t vc1_free_slots_raw[4] = {0, 0, 0, 0};
                    read_free_slots_group<vc1_forwarding, vc1_begin>(vc1_free_slots_raw);
                    visit_forwarding_channels<
                        vc1_far_ring,
                        vc1_begin,
                        vc1_forwarding,
                        vc1_begin,
                        ENABLE_FIRST_LEVEL_ACK_VC1,
                        /*COMBINED_TEST=*/true>(vc1_free_slots_raw, far_vc1, state);
                }
            }

            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                uint64_t loop_end_cycles = get_timestamp();
                uint64_t loop_delta_cycles = loop_end_cycles - loop_start_cycles;
                update_bw_cycles(loop_delta_cycles, tx_progress, false, local_fabric_telemetry);
            }
            if constexpr (FABRIC_TELEMETRY_ANY_DYNAMIC_STAT) {
                update_telemetry(
                    local_sender_channel_free_slots_stream_ids,
                    tx_progress,
                    false,
                    local_fabric_telemetry,
                    fabric_telemetry);
            }

            if ((++fabric_heartbeat_counter & 0x3F) == 0) {
                *fabric_heartbeat_ptr = 0xDCBA0000 | fabric_heartbeat_counter;
            }

            if constexpr (enable_context_switch) {
                // shouldn't do noc counter sync since we are not incrementing them
                if constexpr (IDLE_CONTEXT_SWITCHING) {
                    if (state.attempted || packets_sent() != round_packets_sent) {
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

            close_perf_recording_window(perf_telemetry_recorder);
            if constexpr (perf_telemetry_mode != PerfTelemetryRecorderType::NONE) {
                if (captured_an_event(perf_telemetry_recorder) ||
                    any_sender_channels_active(local_sender_channel_free_slots_stream_ids)) {
                    write_perf_recording_window_results(perf_telemetry_recorder, local_perf_telemetry_buffer);
                }
            }
        }
    };

    while (!got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
        switch (reinterpret_cast<volatile tt_l1_ptr RouterStateManager*>(state_manager_l1)->command) {
            case RouterCommand::RUN:
                state_manager_l1->state = RouterState::RUNNING;
                execute_main_loop();
                break;
            case RouterCommand::PAUSE:
                coordinated_context_switch_start_as_master(termination_signal_ptr);
                execute_pause_command(state_manager_l1, termination_signal_ptr);
                coordinated_context_switch_finish_as_master(termination_signal_ptr);
                break;
            default: ASSERT(false); break;
        }
    }
}

}  // namespace e0

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

// copy the sender_channel_free_slots_stream_ids (in L1) to local memory for performance.
void populate_local_sender_channel_free_slots_stream_id_ordered_map(
    uint32_t has_downstream_edm_vc0_buffer_connection,
    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids) {
    // The whole fabric table is copied, not just this router's compact count, since positions are
    // fabric-flat and a narrower router would otherwise truncate the tail.
    local_sender_channel_free_slots_stream_ids = sender_channel_free_slots_stream_ids;
}

constexpr bool IS_TEARDOWN_MASTER() { return MY_ERISC_ID == 0; }

// Ethernet keepalive: the TX queues' sequence-update timeout (ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, in cycles). The
// teardown master sets it on the sender and receiver TX queues at init and restores the firmware's values at teardown;
// 0 leaves the firmware's value (8000). Taken from the host's TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT until it becomes a host
// setting.
constexpr uint32_t eth_txq_seq_update_timeout = get_named_compile_time_arg_val("EXP_SEQ_UPDATE_TIMEOUT");
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

#if defined(ARCH_BLACKHOLE)
// ERISC 1's whole program in two-ERISC mode: kernel_main's init steps that concern a receiver ERISC, in the same order
// and with the same ERISC syncs (6 in init, 3 in teardown), then the receive program and teardown. ERISC 0 owns the
// sender-side state (sender channels and their worker interfaces, the far receiver's channel view, worker location
// info) and initializes it; ERISC 1 does not touch it. ERISC 1 sets up its downstream router connections itself (the
// same writes, in the same order, as kernel_main's RouterToRouterSender adapters) and forwards through them.
template <bool UNUSED = true>
FORCE_INLINE void e1_kernel_main() {
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

    // Telemetry's TX heartbeat reads the sender channels' free-slot registers (fabric-flat table).
    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> local_sender_channel_free_slots_stream_ids =
        sender_channel_free_slots_stream_ids;

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::DOWNSTREAM_EDM_SETUP_STARTED);
#endif
    std::array<E1DownstreamCursor, E1Downstreams<0>::size> e1_downstreams_vc0;
#if defined(FABRIC_2D_VC1_ACTIVE)
    std::array<E1DownstreamCursor, E1Downstreams<1>::size> e1_downstreams_vc1;
#endif
    const E1DownstreamSetup e1_downstream_vcs[] = {
        e1_downstream_setup<0>(
            e1_downstreams_vc0,
            has_downstream_edm_vc0_buffer_connection,
            downstream_edm_vc0_buffer_base_addresses.data(),
            downstream_edm_vc0_worker_location_info_addresses.data(),
            downstream_edm_vc0_worker_registration_ids.data(),
            local_sem_for_teardown_from_downstream_edm.data(),
            downstream_edm_vc0_noc_x,
            downstream_edm_vc0_noc_y),
#if defined(FABRIC_2D_VC1_ACTIVE)
        e1_downstream_setup<1>(
            e1_downstreams_vc1,
            has_downstream_edm_vc1_buffer_connection,
            downstream_edm_vc1_buffer_base_addresses.data(),
            downstream_edm_vc1_worker_location_info_addresses.data(),
            downstream_edm_vc1_worker_registration_ids.data(),
            local_sem_for_teardown_from_downstream_edm.data() + NUM_DOWNSTREAM_SENDERS_VC0,
            downstream_edm_vc1_noc_x,
            downstream_edm_vc1_noc_y),
#endif
    };
    constexpr uint32_t e1_num_downstream_vcs = sizeof(e1_downstream_vcs) / sizeof(e1_downstream_vcs[0]);
    e1_init_downstreams(e1_downstream_vcs, e1_num_downstream_vcs);
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

    e1_open_downstreams(e1_downstream_vcs, e1_num_downstream_vcs);

    wait_for_other_local_erisc();
#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::VCS_OPENED);
#endif
#if defined(FABRIC_2D)
    constexpr bool vc0_stateful_forwarding = use_2d_single_live_stateful_noc<VC0_DOWNSTREAM_EDM_SIZE>();
#else
    constexpr bool vc0_stateful_forwarding = false;
#endif
    e1_setup_downstream_cmd_bufs(e1_downstream_vcs, e1_num_downstream_vcs, vc0_stateful_forwarding);

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

    run_e1_receive_program(
        e1_downstreams_vc0,
#if defined(FABRIC_2D_VC1_ACTIVE)
        e1_downstreams_vc1,
#endif
        receiver_channel_0_trid_tracker,
#if defined(FABRIC_2D_VC1_ACTIVE)
        receiver_channel_1_trid_tracker,
#endif
        termination_signal_ptr,
        local_sender_channel_free_slots_stream_ids);
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
#endif

void kernel_main() {
#if defined(ARCH_BLACKHOLE)
    // In two-ERISC mode ERISC 1 runs its own program; everything below is the program of every other build.
    if constexpr (e1_receive_program) {
        e1_kernel_main();
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

    populate_local_sender_channel_free_slots_stream_id_ordered_map(
        has_downstream_edm_vc0_buffer_connection, local_sender_channel_free_slots_stream_ids);

    // Receiver-side setup (downstream router adapters, receiver channel buffers) belongs to the ERISC that services the
    // receiver channels. The sending ERISC of a two-ERISC router (e0::runs) skips it: its receiving ERISC has the same
    // compile-time and runtime args and does the same writes in the same init phase, before the next ERISC sync.
    constexpr bool sets_up_receiver_side = !e0::runs;
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
        // initialize the remote receiver channel buffers
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
    if constexpr (e0::runs) {
        e0::wait_for_static_connections(local_sender_channel_worker_interfaces, termination_signal_ptr);
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
    // The sending ERISC of a two-ERISC router runs its own program (namespace e0); every other build the generic loop.
    if constexpr (e0::runs) {
        e0::run_main_loop(local_sender_channel_worker_interfaces, local_sender_channel_free_slots_stream_ids);
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
