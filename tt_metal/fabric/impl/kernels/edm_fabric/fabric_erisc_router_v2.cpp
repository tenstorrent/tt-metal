// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_erisc_microbench.hpp"
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
using LowLatencyFields = tt::tt_fabric::RoutingFieldsConstants::LowLatency;

enum class CoordinatedEriscContextSwitchState : uint32_t {
    NORMAL_EXECUTION = 0,
    RETRAIN_INTENT = 1,
    INTENT_ACK = 2,
    RETRAIN_COMPLETE = 3,
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

template <uint32_t STREAM_ID>
FORCE_INLINE void init_stream_reg_if_used(uint32_t value) {
    if constexpr (STREAM_ID != tt::tt_fabric::k_unused_stream_id) {
        init_ptr_val<STREAM_ID>(value);
    }
}

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
        return sender_channel_free_slots_stream_ids[my_direction];
    } else {
        return sender_channel_free_slots_stream_ids[(1 + my_direction)];
    }
}

constexpr uint32_t get_vc1_downstream_sender_channel_free_slots_stream_id(uint32_t compact_index) {
    auto ds_edm_direction = edm_index_to_edm_direction[my_direction][compact_index];
    if (my_direction > ds_edm_direction) {
        return sender_channel_free_slots_stream_ids[VC1_FABRIC_POSITION_START - 1 + my_direction];
    } else {
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

bool did_something;

FabricPhaseProfiler<
    phase_profile_mode,
    phase_profile_buffer_addr,
    MY_ERISC_ID,
    phase_profile_mask,
    phase_profile_iter_timing,
    phase_profile_txq_mask>
    phase_profiler;

template <size_t N>
constexpr uint32_t count_serviced(const std::array<bool, N>& serviced) {
    uint32_t count = 0;
    for (bool s : serviced) {
        count += s ? 1 : 0;
    }
    return count;
}

FORCE_INLINE void record_sender_channel_usage(size_t sender_channel_index, size_t packet_size_bytes) {
    if constexpr (ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE) {
        channel_trimming_usage_recorder.set_sender_channel_used(sender_channel_index);
        channel_trimming_usage_recorder.update_sender_channel_packet_size(
            sender_channel_index, static_cast<uint16_t>(packet_size_bytes));
    }
}

template <
    uint8_t sender_channel_index,
    uint8_t to_receiver_pkts_sent_id,
    bool SKIP_CONNECTION_LIVENESS_CHECK,
    bool SKIP_TAIL = false,
    uint32_t REMOTE_RECEIVER_BASE_ADDR = 0,
    typename SenderChannelT,
    typename WorkerInterfaceT,
    typename ReceiverPointersT>
FORCE_INLINE void send_next_data(
    SenderChannelT& sender_buffer_channel,
    WorkerInterfaceT& sender_worker_interface,
    ReceiverPointersT& outbound_to_receiver_channel_pointers,
    PerfTelemetryRecorder& perf_telemetry_recorder,
    uint32_t exp_payload_size_bytes = 0) {
    auto& remote_receiver_num_free_slots = outbound_to_receiver_channel_pointers.num_free_slots;
    uint32_t src_addr = sender_buffer_channel.get_cached_next_buffer_slot_addr();

    volatile auto* pkt_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(src_addr);
    size_t payload_size_bytes;
    if constexpr (fabric_exp_tx_send_early_size_read) {
        payload_size_bytes = exp_payload_size_bytes;
    } else {
        payload_size_bytes = pkt_header->get_payload_size_including_header();
    }

    record_sender_channel_usage(sender_channel_index, payload_size_bytes);

    const auto dest_addr = outbound_to_receiver_channel_pointers.remote_receiver_channel_address_ptr;

    if constexpr (!skip_src_ch_id_update) {
        pkt_header->src_ch_id = sender_channel_index;
    }

    if constexpr (ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA) {
        const uint32_t prespin_start = phase_profiler.open<PHASE_TX_TXQ_PRESPIN>();
        while (internal_::eth_txq_is_busy(sender_txq_id)) {
        }
        phase_profiler.close<PHASE_TX_TXQ_PRESPIN>(prespin_start);
    }
    internal_::eth_send_packet_bytes_unsafe(sender_txq_id, src_addr, dest_addr, payload_size_bytes);

    if constexpr (!fabric_exp_tx_send_no_write_counter) {
        sender_worker_interface.template update_write_counter_for_send<SKIP_CONNECTION_LIVENESS_CHECK>();
    }

    if constexpr (fabric_exp_tx_lean_vc0_loop && REMOTE_RECEIVER_BASE_ADDR != 0) {
        outbound_to_receiver_channel_pointers
            .template advance_remote_receiver_buffer_pointer_ct<REMOTE_RECEIVER_BASE_ADDR, channel_buffer_size>();
    } else {
        outbound_to_receiver_channel_pointers.advance_remote_receiver_buffer_pointer();
    }

    if constexpr (fabric_exp_tx_send_stride_slot) {
        if constexpr (fabric_exp_tx_lean_vc0_loop) {
            constexpr size_t sender_entry_idx = SENDER_TO_ENTRY_IDX[sender_channel_index];
            constexpr size_t sender_base_addr = channel_allocs::template Entry<sender_entry_idx>::base_address;
            sender_buffer_channel
                .template advance_to_next_cached_buffer_slot_addr_by_stride_ct<sender_base_addr, channel_buffer_size>();
        } else {
            sender_buffer_channel.advance_to_next_cached_buffer_slot_addr_by_stride();
        }
    } else {
        sender_buffer_channel.advance_to_next_cached_buffer_slot_addr();
    }
    record_packet_send(perf_telemetry_recorder, sender_channel_index, payload_size_bytes);
    if constexpr (SKIP_TAIL) {
        return;
    }
    remote_receiver_num_free_slots--;

    const uint32_t spin_start = phase_profiler.open<PHASE_TX_TXQ_SPIN>();
    while (internal_::eth_txq_is_busy(sender_txq_id)) {
    }
    phase_profiler.close<PHASE_TX_TXQ_SPIN>(spin_start);
    remote_update_ptr_val<to_receiver_pkts_sent_id, sender_txq_id>(1U);
}

FORCE_INLINE bool exp_local_delivery_is_data_only(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start, tt::tt_fabric::NocSendType noc_send_type) {
    if (noc_send_type == tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE) {
        return true;
    }
    if (noc_send_type == tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE) {
        const auto& scatter = packet_start->command_fields.unicast_scatter_write;
        const uint8_t last_encoding = (scatter.chunk_encoding >> (2 * (scatter.chunk_count - 1))) & 0b11;
        return last_encoding == tt::tt_fabric::NocScatterWriteChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE;
    }
    return false;
}

template <size_t N>
constexpr bool exp_stream_unused_in(const std::array<uint32_t, N>& ids, uint32_t id) {
    for (size_t i = 0; i < N; i++) {
        if (ids[i] == id) {
            return false;
        }
    }
    return true;
}

#if defined(FABRIC_DRISC_OFFLOAD)
static_assert(
    drisc_ring_base_addr + drisc_ring_slots * channel_buffer_size <= exp_push_drisc_l1_base + exp_push_drisc_l1_size,
    "DRISC 16-slot staging ring exceeds DRISC UNRESERVED L1");
static_assert(
    exp_stream_unused_in(sender_channel_free_slots_stream_ids, drisc_free_slots_stream_id) &&
        exp_stream_unused_in(to_receiver_packets_sent_streams, drisc_free_slots_stream_id) &&
        exp_stream_unused_in(to_sender_packets_acked_streams, drisc_free_slots_stream_id) &&
        exp_stream_unused_in(to_sender_packets_completed_streams, drisc_free_slots_stream_id),
    "FABRIC_DRISC_OFFLOAD free-slots stream register is in use on ERISC");

uint32_t drisc_offload_next_slot_addr;

FORCE_INLINE void drisc_offload_init_noc_state() {
    constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
    noc_inline_dw_write_set_state</*posted=*/true, /*set_val=*/true>(
        get_noc_addr(
            exp_push_drisc_x,
            exp_push_drisc_y,
            STREAM_REG_ADDR(drisc_doorbell_stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX),
            noc),
        1u << REMOTE_DEST_BUF_WORDS_FREE_INC,
        0xF,
        read_cmd_buf,
        noc,
        tt::tt_fabric::forward_and_local_write_noc_vc);
    noc_async_write_one_packet_with_trid_set_state</*posted=*/false>(
        get_noc_addr(exp_push_drisc_x, exp_push_drisc_y, drisc_ring_base_addr, noc),
        tt::tt_fabric::local_chip_data_cmd_buf,
        noc,
        tt::tt_fabric::forward_and_local_write_noc_vc);
}

FORCE_INLINE void drisc_offload_push_packet(uint32_t pkt_addr, uint32_t total_bytes, uint8_t trid) {
    constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
    constexpr uint8_t data_cmd_buf = tt::tt_fabric::local_chip_data_cmd_buf;
    const uint32_t dst_addr = drisc_offload_next_slot_addr;
    const uint32_t next_addr = dst_addr + channel_buffer_size;
    drisc_offload_next_slot_addr = next_addr == drisc_ring_end_addr ? drisc_ring_base_addr : next_addr;

    while (!noc_cmd_buf_ready(noc, data_cmd_buf)) {
    }
    NOC_CMD_BUF_WRITE_REG(noc, data_cmd_buf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(trid));
    ncrisc_noc_write_with_state<
        DM_DEDICATED_NOC,
        /*posted=*/false,
        /*update_counter=*/true,
        /*one_packet=*/false,
        /*skip_cmdbuf_chk=*/true>(noc, data_cmd_buf, pkt_addr, dst_addr, total_bytes);
    noc_inline_dw_write_with_state<
        /*update_addr_lo=*/false,
        /*update_counter=*/true,
        /*posted=*/true,
        /*update_addr_hi=*/false,
        /*update_val=*/false,
        InlineWriteDst::REG>(0, 0, read_cmd_buf, noc);
}
#endif

FORCE_INLINE void generic_rx_local_delivery_impl(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
    uint16_t payload_size_bytes,
    tt::tt_fabric::NocSendType noc_send_type,
    uint8_t transaction_id,
    uint8_t rx_channel_id) {
    const uint32_t local_start = phase_profiler.open<PHASE_RX_LOCAL>();
    execute_chip_unicast_to_local_chip_impl(
        packet_start, payload_size_bytes, noc_send_type, transaction_id, rx_channel_id);
    phase_profiler.close<PHASE_RX_LOCAL>(local_start);
}

FORCE_INLINE void generic_rx_local_delivery(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
    uint16_t payload_size_bytes,
    uint8_t transaction_id,
    uint8_t rx_channel_id) {
    auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_start);
    generic_rx_local_delivery_impl(
        packet_start, payload_size_bytes, packed.noc_send_type, transaction_id, rx_channel_id);
}

template <typename DownstreamSenderT>
FORCE_INLINE bool can_forward_packet_completely(
    ROUTING_FIELDS_TYPE cached_routing_fields, DownstreamSenderT& downstream_edm_interface) {
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
    return map_downstream_direction_to_compact_index(downstream_direction);
}

template <size_t DOWNSTREAM_EDM_SIZE, eth_chan_directions DIRECTION>
constexpr bool dispatch_arm_is_realizable() {
    return get_downstream_edm_interface_index<DIRECTION>() < DOWNSTREAM_EDM_SIZE;
}

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
template <uint8_t rx_channel_id, typename DownstreamSenderT>
FORCE_INLINE void receiver_forward_packet(
    tt_l1_ptr PACKET_HEADER_TYPE* packet_start,
    ROUTING_FIELDS_TYPE cached_routing_fields,
    DownstreamSenderT& downstream_edm_interface,
    uint8_t transaction_id) {
    constexpr bool ENABLE_STATEFUL_NOC_APIS =
#if !defined(DEBUG_PRINT_ENABLED) and !defined(WATCHER_ENABLED)
        !FORCE_ALL_PATHS_TO_USE_SAME_NOC && true;
#else
        false;
#endif
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::RoutingFields>) {
        bool start_distance_is_terminal_value =
            (cached_routing_fields.value & tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK) ==
            tt::tt_fabric::RoutingFields::LAST_HOP_DISTANCE_VAL;
        uint16_t payload_size_bytes = packet_start->payload_size_bytes;
        bool not_last_destination_device = cached_routing_fields.value != tt::tt_fabric::RoutingFields::LAST_MCAST_VAL;
        if (not_last_destination_device) {
            const uint32_t forward_start = phase_profiler.open<PHASE_RX_FORWARD>();
            forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                packet_start, payload_size_bytes, cached_routing_fields, downstream_edm_interface, transaction_id);
            channel_trimming_usage_recorder.set_sender_channel_forwarded_to(rx_channel_id, 1);
            phase_profiler.close<PHASE_RX_FORWARD>(forward_start);
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
                const uint32_t forward_start = phase_profiler.open<PHASE_RX_FORWARD>();
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                    packet_start, payload_size_bytes, cached_routing_fields, downstream_edm_interface, transaction_id);
                phase_profiler.close<PHASE_RX_FORWARD>(forward_start);
                channel_trimming_usage_recorder.set_sender_channel_forwarded_to(rx_channel_id, 1);
            } break;
            case LowLatencyFields::WRITE_AND_FORWARD: {
                const auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_start);
                if (packed.noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE) [[unlikely]] {
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
                    const uint32_t forward_start = phase_profiler.open<PHASE_RX_FORWARD>();
                    forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                        packet_start,
                        payload_size_bytes,
                        cached_routing_fields,
                        downstream_edm_interface,
                        transaction_id);
                    phase_profiler.close<PHASE_RX_FORWARD>(forward_start);
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
template <uint8_t rx_channel_id, typename LocalRelayInterfaceT>
FORCE_INLINE void forward_to_local_destination(
    LocalRelayInterfaceT& local_relay_interface,
    tt_l1_ptr PACKET_HEADER_TYPE* packet_start,
    uint16_t payload_size_bytes,
    uint8_t transaction_id) {
    if constexpr (udm_mode) {
        execute_chip_unicast_to_relay(
            local_relay_interface, packet_start, payload_size_bytes, transaction_id, rx_channel_id);
    } else {
        generic_rx_local_delivery(packet_start, payload_size_bytes, transaction_id, rx_channel_id);
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
    return is_ew && LIVE == 0b0001 && NUM_ACTIVE_ERISCS == 1 && !FORCE_ALL_PATHS_TO_USE_SAME_NOC && !udm_mode &&
           !is_intermesh_router && !is_intermesh_router_on_edge && !is_intramesh_router_on_edge;
#else
    return false;
#endif
}

template <typename DownstreamSenderT, typename LocalRelayInterfaceT, size_t DOWNSTREAM_EDM_SIZE>
FORCE_INLINE __attribute__((optimize("no-jump-tables"))) bool admit_2d_dispatch(
    uint8_t action,
    bool ld,
    std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    LocalRelayInterfaceT& local_relay_interface) {
    constexpr uint8_t LIVE = live_eth_mask_2d<DOWNSTREAM_EDM_SIZE>();
    constexpr auto dirs = Routing2DCodec::fwd_dirs<static_cast<eth_chan_directions>(my_direction)>();
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
        if (get_ptr_val(local_sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(i)]) !=
            static_cast<int32_t>(SENDER_NUM_BUFFERS_ARRAY[i])) {
            return true;
        }
    }
    return false;
}

constexpr bool IS_RETRAIN_SYNC_MASTER() { return enable_context_switch; }

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
// Sender-channel grouped free-slot polling & idle-channel helpers (pure C++)
// ============================================================================
constexpr size_t exp_grouped_packet_check_channels = ACTUAL_VC0_SENDER_CHANNELS < 4 ? ACTUAL_VC0_SENDER_CHANNELS : 4;

template <size_t CH>
constexpr uint32_t exp_free_slots_reg_addr() {
    return STREAM_REG_ADDR(
        sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(CH)],
        STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
}

template <size_t CH>
FORCE_INLINE uint32_t exp_read_free_slots_raw() {
    return *reinterpret_cast<volatile uint32_t*>(exp_free_slots_reg_addr<CH>());
}

template <size_t N = exp_grouped_packet_check_channels>
FORCE_INLINE void exp_read_grouped_free_slots(uint32_t (&raw)[4]) {
    if constexpr (N >= 1) {
        raw[0] = exp_read_free_slots_raw<0>();
    }
    if constexpr (N >= 2) {
        raw[1] = exp_read_free_slots_raw<1>();
    }
    if constexpr (N >= 3) {
        raw[2] = exp_read_free_slots_raw<2>();
    }
    if constexpr (N >= 4) {
        raw[3] = exp_read_free_slots_raw<3>();
    }
    // Compiler scheduling barrier: keeps all N stream-register loads back-to-back before ch0 masks raw[0].
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

constexpr bool exp_skip_idle_channel(size_t ch) {
    return ch >= 1 && ch < exp_grouped_packet_check_channels && is_sender_channel_serviced[ch];
}
constexpr bool exp_skip_idle_supported = [] {
    for (size_t ch = 1; ch <= (is_2d_fabric ? 3 : 1) && ch < MAX_NUM_SENDER_CHANNELS; ch++) {
        if (is_sender_channel_serviced[ch] && !exp_skip_idle_channel(ch)) {
            return false;
        }
    }
    bool any = false;
    for (size_t ch = 1; ch < exp_grouped_packet_check_channels; ch++) {
        if (is_sender_channel_serviced[ch]) {
            if (!sender_ch_live_check_skip[ch]) {
                return false;
            }
            any = true;
        }
    }
    return any && fabric_exp_tx_lean_completions && fabric_exp_tx_grouped_packet_check && !ENABLE_FIRST_LEVEL_ACK_VC0 &&
           !fabric_exp_tx_amortize_worker_credits && !fabric_exp_tx_upstream_on_e1;
}();
constexpr bool fabric_exp_tx_skip_idle_per_channel =
    (fabric_exp_flags & EXP_TX_SKIP_IDLE_PER_CHANNEL) != 0 && exp_skip_idle_supported;

constexpr bool exp_skip_idle_bit_test = [] {
    size_t n = 0;
    for (size_t ch = 1; ch < exp_grouped_packet_check_channels; ch++) {
        if (exp_skip_idle_channel(ch)) {
            const size_t slots = SENDER_NUM_BUFFERS_ARRAY[ch];
            if ((slots & (slots - 1)) != 0 || slots >= (1u << REMOTE_DEST_WORDS_FREE_WIDTH) || (n != 0 && slots != n)) {
                return false;
            }
            n = slots;
        }
    }
    return n != 0;
}();

template <size_t CH>
FORCE_INLINE bool exp_channel_all_slots_free(uint32_t raw) {
    if constexpr (exp_skip_idle_bit_test) {
        return (raw & SENDER_NUM_BUFFERS_ARRAY[CH]) != 0;
    } else {
        return (raw & ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1)) == SENDER_NUM_BUFFERS_ARRAY[CH];
    }
}

// ============================================================================
// Sender & Receiver Channel Steps (Pure C++, Zero asm)
// ============================================================================
template <
    uint8_t sender_channel_index,
    uint8_t to_receiver_pkts_sent_id,
    bool SKIP_CONNECTION_LIVENESS_CHECK,
    bool enable_first_level_ack,
    uint8_t VC_RECEIVER_CHANNEL = 0,
    typename SenderChannelT,
    typename WorkerInterfaceT,
    typename ReceiverPointersT,
    typename ReceiverChannelT,
    typename LocalTelemetryT,
    typename SenderChannelFromReceiverCreditsT>
#if !defined(FABRIC_2D_VC1_ACTIVE)
FORCE_INLINE
#endif
    bool
    run_sender_channel_step_impl(
        SenderChannelT& local_sender_channel,
        WorkerInterfaceT& local_sender_channel_worker_interface,
        ReceiverPointersT& outbound_to_receiver_channel_pointers,
        ReceiverChannelT& remote_receiver_channel,
        bool& channel_connection_established,
        uint32_t sender_channel_free_slots_stream_id,
        SenderChannelFromReceiverCreditsT& sender_channel_from_receiver_credits,
        PerfTelemetryRecorder& perf_telemetry_recorder,
        LocalTelemetryT& local_fabric_telemetry,
        uint32_t exp_grouped_free_slots_raw = 0) {
    bool progress = false;
    constexpr bool use_bubble_flow_control =
        sender_channel_is_traffic_injection_channel[sender_channel_index] && enable_deadlock_avoidance;
    static_assert(
        !use_bubble_flow_control || enable_first_level_ack,
        "Bubble flow control and first level ack must be set to the same values");

    if constexpr (fabric_exp_tx_skip_idle_per_channel && exp_skip_idle_channel(sender_channel_index)) {
        if (exp_channel_all_slots_free<sender_channel_index>(exp_grouped_free_slots_raw) &&
            !sender_channel_from_receiver_credits.has_completions_pending()) {
            return false;
        }
    }

    bool receiver_has_space_for_packet;
    if constexpr (use_bubble_flow_control) {
        receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.num_free_slots >=
                                        BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS;
    } else {
        receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet();
    }
    uint32_t free_slots;
    if constexpr (fabric_exp_tx_grouped_packet_check && sender_channel_index < exp_grouped_packet_check_channels) {
        free_slots = exp_grouped_free_slots_raw & ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1);
    } else {
        free_slots = get_ptr_val(sender_channel_free_slots_stream_id);
    }
    bool has_unsent_packet = free_slots != WorkerInterfaceT::num_buffers;
    bool can_send = receiver_has_space_for_packet && has_unsent_packet;

    uint32_t exp_payload_size_bytes = 0;
    if constexpr (fabric_exp_tx_send_early_size_read) {
        if (can_send) {
            exp_payload_size_bytes = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
                                         local_sender_channel.get_cached_next_buffer_slot_addr())
                                         ->get_payload_size_including_header();
            if constexpr (!ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA && !fabric_exp_tx_no_send_txq_check) {
                can_send = !internal_::eth_txq_is_busy(sender_txq_id);
            }
        }
    } else if constexpr (!ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA && !fabric_exp_tx_no_send_txq_check) {
        can_send = can_send && !internal_::eth_txq_is_busy(sender_txq_id);
    }
    constexpr bool exp_overlap_send_completion =
#if defined(ARCH_BLACKHOLE)
        fabric_exp_tx_send_overlap_completion && sender_channel_uses_counter_credits(sender_channel_index) &&
        !ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA && !fabric_exp_tx_send_early_txq_read && !ENABLE_RISC_CPU_DATA_CACHE;
#else
        false;
#endif
    int32_t completions_since_last_check = 0;
    if (can_send) {
        did_something = true;
        progress = true;
        phase_profiler.mark_progress();
        const uint32_t send_start = phase_profiler.open<PHASE_TX_SEND>();

        auto* pkt_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(
            local_sender_channel.get_cached_next_buffer_slot_addr());
        constexpr size_t rem_rx_entry_idx =
            eth_remote_channel_allocs::receiver_channel_to_entry_index[VC_RECEIVER_CHANNEL];
        constexpr uint32_t rem_rx_base_addr =
            static_cast<uint32_t>(eth_remote_channel_allocs::template Entry<rem_rx_entry_idx>::base_address);
        send_next_data<
            sender_channel_index,
            to_receiver_pkts_sent_id,
            SKIP_CONNECTION_LIVENESS_CHECK,
            exp_overlap_send_completion,
            rem_rx_base_addr>(
            local_sender_channel,
            local_sender_channel_worker_interface,
            outbound_to_receiver_channel_pointers,
            perf_telemetry_recorder,
            exp_payload_size_bytes);
        if constexpr (exp_overlap_send_completion) {
            constexpr uint32_t txq_base = ETH_TXQ0_REGS_START + sender_txq_id * ETH_TXQ_REGS_SIZE;
            constexpr uint32_t rem_stream_addr =
                STREAM_REG_ADDR(to_receiver_pkts_sent_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
            volatile uint32_t* const txq_regs = reinterpret_cast<volatile uint32_t*>(txq_base);
            volatile tt_l1_ptr uint32_t* const cptr =
                sender_channel_from_receiver_credits.completions_received_counter_ptr;

            // Issue the TX-queue busy reads (BH-55 dummy CMD + STATUS) and the L1 completion-counter read
            // back to back, fenced so the K=6 address/counter instructions below stay in the load shadow.
            std::atomic_signal_fence(std::memory_order_seq_cst);
            (void)txq_regs[ETH_TXQ_CMD >> 2];
            const uint32_t txq_status_raw = txq_regs[ETH_TXQ_STATUS >> 2];
            const uint32_t comp_raw = *cptr;
            std::atomic_signal_fence(std::memory_order_seq_cst);

            outbound_to_receiver_channel_pointers.num_free_slots--;
            if constexpr (fabric_exp_tx_lean_vc0_loop || fabric_exp_tx_send_ct_free_slot_stream) {
                increment_local_update_ptr_val<
                    sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(sender_channel_index)]>(1);
            } else {
                increment_local_update_ptr_val(sender_channel_free_slots_stream_id, 1);
            }
            std::atomic_signal_fence(std::memory_order_seq_cst);

            if (static_cast<int32_t>(txq_status_raw << 15) < 0) [[unlikely]] {
                while (internal_::eth_txq_is_busy(sender_txq_id)) {
                }
            }
            txq_regs[ETH_TXQ_DEST_ADDR >> 2] = rem_stream_addr;
            txq_regs[ETH_TXQ_REMOTE_REG_DATA >> 2] = 1u << REMOTE_DEST_BUF_WORDS_FREE_INC;
            txq_regs[ETH_TXQ_CMD >> 2] = ETH_TXQ_CMD_START_REG;
            if constexpr (fabric_exp_tx_lean_completions) {
                sender_channel_from_receiver_credits.note_packet_sent();
            }
            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                update_bw_counters(pkt_header, local_fabric_telemetry);
            }
            completions_since_last_check = static_cast<int32_t>(
                comp_raw - sender_channel_from_receiver_credits.completions_received_and_processed);
        } else {
            if constexpr (fabric_exp_tx_lean_completions) {
                sender_channel_from_receiver_credits.note_packet_sent();
            }
            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                update_bw_counters(pkt_header, local_fabric_telemetry);
            }
            if constexpr (fabric_exp_tx_send_ct_free_slot_stream) {
                increment_local_update_ptr_val<
                    sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(sender_channel_index)]>(1);
            } else {
                increment_local_update_ptr_val(sender_channel_free_slots_stream_id, 1);
            }
        }
        phase_profiler.close<PHASE_TX_SEND>(send_start);
        phase_profiler.count_pkt();
        fabric_experiment_spin<fabric_exp_tx_delay>();
    } else if (has_unsent_packet) {
        phase_profiler.mark_blocked();
        phase_profiler.count_event(receiver_has_space_for_packet ? EVENT_TX_BLOCKED_TXQ : EVENT_TX_BLOCKED_CREDIT);
    }

    if (!exp_overlap_send_completion || !can_send) {
        if (!fabric_exp_tx_lean_completions || sender_channel_from_receiver_credits.has_completions_pending()) {
            completions_since_last_check =
                sender_channel_from_receiver_credits
                    .template get_num_unprocessed_completions_from_receiver<ENABLE_RISC_CPU_DATA_CACHE>();
        }
    }
    if (completions_since_last_check) {
        const uint32_t completion_start = phase_profiler.open<PHASE_TX_COMPLETION>();
        outbound_to_receiver_channel_pointers.num_free_slots += completions_since_last_check;
        sender_channel_from_receiver_credits.increment_num_processed_completions(completions_since_last_check);
        if constexpr (!enable_first_level_ack) {
            send_credits_to_upstream_workers<enable_deadlock_avoidance, SKIP_CONNECTION_LIVENESS_CHECK>(
                local_sender_channel_worker_interface, completions_since_last_check, channel_connection_established);
        }
        phase_profiler.close<PHASE_TX_COMPLETION>(completion_start);
    }

    if constexpr (enable_first_level_ack) {
        uint32_t acks_since_last_check = 0;
        if (!fabric_exp_tx_lean_completions || sender_channel_from_receiver_credits.has_acks_pending()) {
            acks_since_last_check = sender_channel_from_receiver_credits
                                        .template get_num_unprocessed_acks_from_receiver<ENABLE_RISC_CPU_DATA_CACHE>();
        }
        if (acks_since_last_check > 0) {
            const uint32_t ack_start = phase_profiler.open<PHASE_TX_COMPLETION>();
            sender_channel_from_receiver_credits.increment_num_processed_acks(acks_since_last_check);
            send_credits_to_upstream_workers<enable_deadlock_avoidance, SKIP_CONNECTION_LIVENESS_CHECK>(
                local_sender_channel_worker_interface, acks_since_last_check, channel_connection_established);
            phase_profiler.close<PHASE_TX_COMPLETION>(ack_start);
        }
    }

    if constexpr (!SKIP_CONNECTION_LIVENESS_CHECK) {
        auto check_connection_status =
            !channel_connection_established || local_sender_channel_worker_interface.has_worker_teardown_request();
        if (check_connection_status) {
            const uint32_t conn_start = phase_profiler.open<PHASE_TX_CONN>();
            check_worker_connections<MY_ETH_CHANNEL, ENABLE_RISC_CPU_DATA_CACHE>(
                local_sender_channel_worker_interface,
                channel_connection_established,
                sender_channel_free_slots_stream_id);
            phase_profiler.close<PHASE_TX_CONN>(conn_start);
        }
    }
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
        RemoteEthReceiverChannels& remote_receiver_channels,
        std::array<bool, NUM_SENDER_CHANNELS>& channel_connection_established,
        std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids,
        SenderChannelFromReceiverCredits<NUM_SENDER_CHANNELS>& sender_channel_from_receiver_credits,
        PerfTelemetryRecorder& perf_telemetry_recorder,
        LocalTelemetryT& local_fabric_telemetry,
        uint32_t exp_grouped_free_slots_raw = 0) {
    if constexpr (is_sender_channel_serviced[sender_channel_index]) {
        const uint32_t visit_start = phase_profiler.open<PHASE_TX_VISIT>();
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();

        const bool progress = run_sender_channel_step_impl<
            sender_channel_index,
            to_receiver_packets_sent_streams[VC_RECEIVER_CHANNEL],
            sender_ch_live_check_skip[sender_channel_index],
            enable_first_level_ack,
            VC_RECEIVER_CHANNEL>(
            local_sender_channels.template get<sender_channel_index>(),
            local_sender_channel_worker_interfaces.template get<sender_channel_index>(),
            outbound_to_receiver_channel_pointers,
            remote_receiver_channels.template get<VC_RECEIVER_CHANNEL>(),
            channel_connection_established[sender_channel_index],
            local_sender_channel_free_slots_stream_ids[fabric_position_for_compact_sender(sender_channel_index)],
            sender_channel_from_receiver_credits.template get<sender_channel_index>(),
            perf_telemetry_recorder,
            local_fabric_telemetry,
            exp_grouped_free_slots_raw);
        phase_profiler.close<PHASE_TX_VISIT>(visit_start);
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
    LocalTelemetryT& local_fabric_telemetry) {
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

            const uint32_t ack_start = phase_profiler.open<PHASE_RX_ACK>();
            receiver_send_received_ack<ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK>(
                receiver_channel_response_credit_sender, src_ch_id);
            ack_counter.increment();
            phase_profiler.close<PHASE_RX_ACK>(ack_start);
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
        const uint32_t rx_start = phase_profiler.open<PHASE_RX_DECODE_ADMIT>();
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr PACKET_HEADER_TYPE* packet_header = const_cast<PACKET_HEADER_TYPE*>(
            local_receiver_channel.template get_packet_header<PACKET_HEADER_TYPE>(receiver_buffer_index));

#if defined(FABRIC_DRISC_OFFLOAD)
        bool can_push = get_ptr_val<drisc_free_slots_stream_id>() != 0;
        const bool admitted = can_push;
        if constexpr (enable_trid_flush_check_on_noc_txn) {
            can_push = can_push && receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
        }
        if (can_push) {
            const auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_header);
            if constexpr (!skip_src_ch_id_update && !enable_first_level_ack) {
                receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);
            }
            did_something = true;
            progress = true;
            phase_profiler.mark_progress();
            phase_profiler.close<PHASE_RX_DECODE_ADMIT>(rx_start);
            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                update_bw_counters(packet_header, local_fabric_telemetry);
            }
            channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(receiver_channel);
            const uint8_t trid = receiver_channel_trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(
                receiver_buffer_index);
            const uint32_t local_start = phase_profiler.open<PHASE_RX_LOCAL>();
            drisc_offload_push_packet(
                reinterpret_cast<uint32_t>(packet_header),
                packed.payload_size_bytes + sizeof(PACKET_HEADER_TYPE),
                trid);
            phase_profiler.close<PHASE_RX_LOCAL>(local_start);
            increment_local_update_ptr_val<drisc_free_slots_stream_id>(-1);
            wr_sent_counter.increment();
            if constexpr (!enable_first_level_ack) {
                increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);
            }
            phase_profiler.count_pkt();
            fabric_experiment_spin<fabric_exp_rx_delay>();
        } else {
            phase_profiler.mark_blocked();
            phase_profiler.count_event(admitted ? EVENT_RX_BLOCKED_TRID : EVENT_RX_BLOCKED_ADMIT);
        }
#else
        {
            ROUTING_FIELDS_TYPE cached_routing_fields{};
#ifndef FABRIC_2D
            cached_routing_fields = packet_header->routing_fields;
#endif
            if constexpr (!skip_src_ch_id_update && !enable_first_level_ack) {
                receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);
            }
            uint8_t action = 0;
            bool local_deliver = false;
            bool intermesh_egress = false;
            uint8_t intermesh_egress_index = 0;
            bool can_send_to_all_local_chip_receivers;
            if constexpr (is_2d_fabric) {
#if defined(FABRIC_2D)
                if constexpr (receiver_channel_is_intermesh_ingress[receiver_channel]) {
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
                        routing_table.inter_mesh_direction_table.get_original_direction(
                            packet_header->dst_start_mesh_id));
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
            const bool admitted = can_send_to_all_local_chip_receivers;
            if constexpr (enable_trid_flush_check_on_noc_txn) {
                bool trid_flushed = receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
                can_send_to_all_local_chip_receivers &= trid_flushed;
            }
            if (can_send_to_all_local_chip_receivers) {
                did_something = true;
                progress = true;
                phase_profiler.mark_progress();
                phase_profiler.close<PHASE_RX_DECODE_ADMIT>(rx_start);
                if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                    update_bw_counters(packet_header, local_fabric_telemetry);
                }
                channel_trimming_usage_recorder.set_receiver_channel_data_forwarded(receiver_channel);
                if constexpr (!is_receiver_channel_forwarding_disabled[receiver_channel]) {
                    uint8_t trid =
                        receiver_channel_trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(
                            receiver_buffer_index);
                    if constexpr (is_2d_fabric) {
#if defined(FABRIC_2D)
                        if (intermesh_egress) {
                            forward_payload_to_downstream_edm<enable_deadlock_avoidance, false>(
                                packet_header,
                                packet_header->payload_size_bytes,
                                cached_routing_fields,
                                downstream_edm_interfaces[intermesh_egress_index],
                                trid);
                            record_2d_forwarded_slots<receiver_channel>(1u << intermesh_egress_index);
                        } else {
                            const uint32_t forward_start = phase_profiler.open<PHASE_RX_FORWARD>();
                            const uint8_t forwarded_slot_mask = forward_2d_dispatch(
                                action, packet_header, cached_routing_fields, downstream_edm_interfaces, trid);
                            record_2d_forwarded_slots<receiver_channel>(forwarded_slot_mask);
                            if (forwarded_slot_mask != 0) {
                                phase_profiler.close<PHASE_RX_FORWARD>(forward_start);
                            }
                            if (local_deliver) {
                                forward_to_local_destination<receiver_channel>(
                                    local_relay_interface, packet_header, packet_header->payload_size_bytes, trid);
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
                phase_profiler.count_pkt();
                fabric_experiment_spin<fabric_exp_rx_delay>();
            } else {
                phase_profiler.mark_blocked();
                phase_profiler.count_event(admitted ? EVENT_RX_BLOCKED_TRID : EVENT_RX_BLOCKED_ADMIT);
            }
        }
#endif
    }

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
            const uint32_t completion_start = phase_profiler.open<PHASE_RX_COMPLETION>();
            auto receiver_buffer_index = wr_flush_counter.get_buffer_index();
            receiver_send_completion_ack<ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK>(
                receiver_channel_response_credit_sender,
                receiver_channel_pointers.get_src_chan_id(receiver_buffer_index));
            completion_counter.increment();
            phase_profiler.close<PHASE_RX_COMPLETION>(completion_start);
        }
    } else if constexpr (fabric_exp_rx_batch_completion && !enable_first_level_ack) {
        constexpr uint32_t batch_target = RECEIVER_NUM_BUFFERS_ARRAY[receiver_channel] >= 8 ? 4 : 1;
        auto& completion_counter = receiver_channel_pointers.completion_counter;
        const uint32_t uncompleted = completion_counter.distance_behind(wr_sent_counter);
#if defined(FABRIC_DRISC_OFFLOAD)
        const bool should_check_completion =
            uncompleted > batch_target ||
            (!progress &&
             (uncompleted >= batch_target || (uncompleted != 0 && get_ptr_val<drisc_free_slots_stream_id>() ==
                                                                      static_cast<int32_t>(drisc_ring_slots))));
#else
        const bool should_check_completion = uncompleted > batch_target || (uncompleted != 0 && !progress);
#endif
        if (should_check_completion) {
            const uint32_t batch = uncompleted > batch_target ? batch_target : uncompleted;
            const auto base_idx = completion_counter.get_buffer_index();
            const auto last_idx = tt::tt_fabric::BufferIndex{
                tt::tt_fabric::wrap_increment_n<RECEIVER_NUM_BUFFERS_ARRAY[receiver_channel]>(
                    base_idx.get(), batch - 1)};
            bool can_send_completion = receiver_channel_trid_tracker.transaction_flushed(last_idx);
#if !defined(FABRIC_DRISC_OFFLOAD)
            if (!progress && batch > 1) {
                can_send_completion =
                    can_send_completion && receiver_channel_trid_tracker.transaction_flushed(base_idx);
            }
#endif
            if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
                can_send_completion = can_send_completion && !internal_::eth_txq_is_busy(receiver_txq_id);
            }
            if (can_send_completion) {
                const uint32_t completion_start = phase_profiler.open<PHASE_RX_COMPLETION>();
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
                phase_profiler.close<PHASE_RX_COMPLETION>(completion_start);
            }
        }
    } else {
        auto& completion_counter = receiver_channel_pointers.completion_counter;
        bool unflushed_writes = !completion_counter.is_caught_up_to(wr_sent_counter);
        auto receiver_buffer_index = completion_counter.get_buffer_index();
        bool next_trid_flushed = receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
        bool can_send_completion = unflushed_writes && next_trid_flushed;
        if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
            can_send_completion = can_send_completion && !internal_::eth_txq_is_busy(receiver_txq_id);
        }
        if (can_send_completion) {
            const uint32_t completion_start = phase_profiler.open<PHASE_RX_COMPLETION>();
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
            phase_profiler.close<PHASE_RX_COMPLETION>(completion_start);
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
    LocalTelemetryT& local_fabric_telemetry) {
    if constexpr (is_receiver_channel_serviced[receiver_channel]) {
        const uint32_t visit_start = phase_profiler.open<PHASE_RX_VISIT>();
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
            local_fabric_telemetry);
        phase_profiler.close<PHASE_RX_VISIT>(visit_start);
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
    ASSERT(false);
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

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp"

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

    const auto& routing_table =
        *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    auto* state_manager_l1 = const_cast<tt_l1_ptr RouterStateManager*>(&routing_table.state_manager);

    const uint8_t my_mesh_coord_y = routing_table.my_mesh_coord_y;
    const uint8_t my_mesh_coord_x = routing_table.my_mesh_coord_x;
    const uint16_t my_mesh_id = routing_table.my_mesh_id;

    auto outbound_to_receiver_channel_pointers =
        ChannelPointersTuple<OutboundReceiverChannelPointers, REMOTE_RECEIVER_NUM_BUFFERS_ARRAY>::make();
    configure_outbound_to_receiver_channel_pointers(outbound_to_receiver_channel_pointers, remote_receiver_channels);

    auto receiver_channel_pointers = ChannelPointersTuple<ReceiverChannelPointers, RECEIVER_NUM_BUFFERS_ARRAY>::make();

    auto outbound_to_receiver_channel_pointer_ch0 =
        outbound_to_receiver_channel_pointers.template get<VC0_RECEIVER_CHANNEL>();

    auto receiver_channel_pointers_ch0 = receiver_channel_pointers.template get<0>();
    receiver_channel_pointers_ch0.reset();

#if defined(FABRIC_2D_VC1_SERVICED)
    auto outbound_to_receiver_channel_pointer_ch1 =
        outbound_to_receiver_channel_pointers.template get<VC1_RECEIVER_CHANNEL>();

    auto receiver_channel_pointers_ch1 = receiver_channel_pointers.template get<1>();
    receiver_channel_pointers_ch1.reset();
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

    phase_profiler.init<count_serviced(is_sender_channel_serviced), count_serviced(is_receiver_channel_serviced)>();

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
            for (size_t i = 0; i < iterations_between_ctx_switch_and_teardown_checks; i++) {
                if constexpr (super_speedy_mode) {
                    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();

                    if constexpr (is_sender_channel_serviced[0]) {
                        const uint32_t tx_visit_start = phase_profiler.open<PHASE_TX_VISIT>();
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
                        phase_profiler.close<PHASE_TX_VISIT>(tx_visit_start);
                    }

                    if constexpr (is_receiver_channel_serviced[0]) {
#if defined(FABRIC_DRISC_OFFLOAD)
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
                            local_fabric_telemetry);
#else
                        const uint32_t rx_visit_start = phase_profiler.open<PHASE_RX_VISIT>();
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
                        phase_profiler.close<PHASE_RX_VISIT>(rx_visit_start);
#endif
                    }
                } else {
                    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
                    uint32_t exp_free_slots_raw[4] = {0, 0, 0, 0};
                    if constexpr (fabric_exp_tx_grouped_packet_check && any_sender_channel_serviced) {
                        exp_read_grouped_free_slots(exp_free_slots_raw);
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
                        exp_free_slots_raw[0]);
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
                        local_fabric_telemetry);
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
                        local_fabric_telemetry);
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
                        exp_free_slots_raw[1]);
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
                            exp_free_slots_raw[2]);
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
                            exp_free_slots_raw[3]);
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
                        local_fabric_telemetry);
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
                phase_profiler.end_iteration();
            }
            phase_profiler.flush();

            if constexpr (FABRIC_TELEMETRY_BANDWIDTH) {
                uint64_t loop_end_cycles = get_timestamp();
                uint64_t loop_delta_cycles = loop_end_cycles - loop_start_cycles;
                update_bw_cycles(loop_delta_cycles, tx_progress, rx_progress, local_fabric_telemetry);
            }
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
        tuple_for_each(
            local_sender_channel_worker_interfaces.channel_worker_interfaces,
            [&](auto& interface, size_t idx) { establish_static_connection_from_receiver_side(interface, idx); });
    }
}

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

void populate_local_sender_channel_free_slots_stream_id_ordered_map(
    uint32_t has_downstream_edm_vc0_buffer_connection,
    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>& local_sender_channel_free_slots_stream_ids) {
    local_sender_channel_free_slots_stream_ids = sender_channel_free_slots_stream_ids;
}

constexpr bool IS_TEARDOWN_MASTER() { return MY_ERISC_ID == 0; }

constexpr bool sets_seq_update_timeout = fabric_exp_seq_update_timeout != 0 && IS_TEARDOWN_MASTER();
uint32_t firmware_seq_update_timeouts[2];

__attribute__((optimize("Os"))) void exp_set_seq_update_timeouts() {
    if constexpr (sets_seq_update_timeout) {
        firmware_seq_update_timeouts[0] = eth_txq_reg_read(sender_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT);
        eth_txq_reg_write(sender_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, fabric_exp_seq_update_timeout);
        if constexpr (receiver_txq_id != sender_txq_id) {
            firmware_seq_update_timeouts[1] = eth_txq_reg_read(receiver_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT);
            eth_txq_reg_write(receiver_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, fabric_exp_seq_update_timeout);
        }
    }
}

__attribute__((optimize("Os"))) void exp_restore_seq_update_timeouts() {
    if constexpr (sets_seq_update_timeout) {
        eth_txq_reg_write(sender_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, firmware_seq_update_timeouts[0]);
        if constexpr (receiver_txq_id != sender_txq_id) {
            eth_txq_reg_write(receiver_txq_id, ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT, firmware_seq_update_timeouts[1]);
        }
    }
}

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
#if defined(FABRIC_DRISC_OFFLOAD)
        constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
        noc_async_posted_writes_flushed(noc);
        noc_clear_packet_tags(noc);
        noc_init_one(noc, MEM_NOC_ATOMIC_RET_VAL_ADDR);
        noc_async_write_one_packet(
            reinterpret_cast<uint32_t>(termination_signal_ptr),
            get_noc_addr(exp_push_drisc_x, exp_push_drisc_y, drisc_termination_signal_l1_addr, noc),
            sizeof(uint32_t),
            noc);
        noc_async_write_barrier(noc);
#endif
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

#if defined(FABRIC_DRISC_OFFLOAD)
    if constexpr (is_receiver_channel_serviced[0]) {
        while (get_ptr_val<drisc_free_slots_stream_id>() < static_cast<int32_t>(drisc_ring_slots + 1)) {
        }
        init_ptr_val<drisc_free_slots_stream_id>(0);
    }
#endif

#ifdef ARCH_BLACKHOLE
    noc_clear_packet_tags(noc_index);
#endif

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }
    exp_restore_seq_update_timeouts();
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

void initialize_fabric_telemetry() {
    volatile tt_l1_ptr FabricTelemetry* fabric_telemetry =
        reinterpret_cast<volatile tt_l1_ptr FabricTelemetry*>(eth_l1_mem::address_map::AERISC_FABRIC_TELEMETRY_ADDR);
    const auto* routing_table_l1 = reinterpret_cast<tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(ROUTING_TABLE_BASE);
    memset(const_cast<FabricTelemetry*>(fabric_telemetry), 0, sizeof(FabricTelemetry));

    fabric_telemetry->static_info.mesh_id = routing_table_l1->my_mesh_id;
    fabric_telemetry->static_info.device_id = routing_table_l1->my_device_id;
    fabric_telemetry->static_info.direction = static_cast<uint8_t>(my_direction);
    fabric_telemetry->static_info.version = FABRIC_TELEMETRY_VERSION;
    fabric_telemetry->static_info.fabric_config = 0;
    fabric_telemetry->static_info.neighbor_mesh_id = 0xFFFF;
    fabric_telemetry->static_info.neighbor_device_id = 0xFF;
    fabric_telemetry->static_info.supported_stats = static_cast<DynamicStatistics>(
        DynamicStatistics::BANDWIDTH | DynamicStatistics::ROUTER_STATE | DynamicStatistics::HEARTBEAT_TX |
        DynamicStatistics::HEARTBEAT_RX);

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        fabric_telemetry->dynamic_info.erisc[MY_ERISC_ID].router_state = RouterState::INITIALIZING;
    } else {
        fabric_telemetry->dynamic_info.erisc[0].router_state = RouterState::INITIALIZING;
        fabric_telemetry->dynamic_info.erisc[1].router_state = RouterState::INITIALIZING;
    }
}

void kernel_main() {
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

    if constexpr (phase_profile_mode >= 3 && any_receiver_channel_serviced) {
        run_fabric_erisc_microbench<
            phase_profile_microbench_results_addr,
            phase_profile_microbench_scratch_addr,
            tt::tt_fabric::edm_to_local_chip_noc,
            tt::tt_fabric::local_chip_data_cmd_buf,
            sender_txq_id>();
    }

    initialize_fabric_telemetry();

    eth_txq_reg_write(sender_txq_id, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, DEFAULT_NUM_ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD);
    exp_set_seq_update_timeouts();
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

    init_ptr_val<to_receiver_packets_sent_streams[0]>(0);
#if defined(FABRIC_DRISC_OFFLOAD)
    init_ptr_val<drisc_free_slots_stream_id>(0);
#endif
    init_stream_reg_if_used<to_sender_packets_acked_streams[0]>(0);
    init_stream_reg_if_used<to_sender_packets_acked_streams[1]>(0);
    init_stream_reg_if_used<to_sender_packets_completed_streams[0]>(0);
    init_stream_reg_if_used<to_sender_packets_completed_streams[1]>(0);
    init_ptr_val<sender_channel_free_slots_stream_ids[0]>(SENDER_NUM_BUFFERS_ARRAY[0]);
    init_ptr_val<sender_channel_free_slots_stream_ids[1]>(SENDER_NUM_BUFFERS_ARRAY[1]);

    if (IS_RETRAIN_SYNC_MASTER()) {
        write_stream_scratch_register<ETH_RETRAIN_LINK_SYNC_STREAM_ID>(
            static_cast<CoordinatedEriscCtxType>(CoordinatedEriscContextSwitchState::NORMAL_EXECUTION));
    }

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }

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
    volatile auto edm_local_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(edm_local_sync_ptr_addr);
    volatile auto edm_status_ptr = reinterpret_cast<volatile tt_l1_ptr tt::tt_fabric::EDMStatus*>(edm_status_ptr_addr);

    size_t arg_idx = 0;
    std::array<size_t, MAX_NUM_SENDER_CHANNELS> local_sender_channel_connection_semaphore_addrs;
    for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++) {
        local_sender_channel_connection_semaphore_addrs[i] = get_arg_val<uint32_t>(arg_idx++);
    }
    std::array<size_t, MAX_NUM_SENDER_CHANNELS> local_sender_channel_connection_buffer_index_ids;
    for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++) {
        local_sender_channel_connection_buffer_index_ids[i] = get_arg_val<uint32_t>(arg_idx++);
    }

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

    std::array<size_t, MAX_NUM_SENDER_CHANNELS> my_sem_for_teardown_from_edm;
    for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++) {
        my_sem_for_teardown_from_edm[i] = get_arg_val<uint32_t>(arg_idx++);
    }

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

    std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> local_sender_channel_free_slots_stream_ids;

    [[maybe_unused]] const auto& local_sem_for_teardown_from_downstream_edm =
        take_first_n_elements<NUM_DOWNSTREAM_CHANNELS, MAX_NUM_SENDER_CHANNELS, size_t>(my_sem_for_teardown_from_edm);

    auto remote_receiver_channels = tt::tt_fabric::ReceiverChannelBuffersFromAllocs<
        PACKET_HEADER_TYPE,
        eth_remote_channel_allocs,
        REMOTE_RECEIVER_TO_ENTRY_IDX>::make();

    auto local_receiver_channels = tt::tt_fabric::
        ReceiverChannelBuffersFromAllocs<PACKET_HEADER_TYPE, channel_allocs, RECEIVER_TO_ENTRY_IDX>::make();

#if !defined(FABRIC_2D_VC2_SERVICED)
    constexpr bool exp_use_lean_stride_sender_channels =
        fabric_exp_tx_lean_vc0_loop && fabric_exp_tx_send_stride_slot && !super_speedy_mode;
#else
    constexpr bool exp_use_lean_stride_sender_channels = false;
#endif
    using SenderChannelsAllocT = std::conditional_t<
        exp_use_lean_stride_sender_channels,
        tt::tt_fabric::LeanSenderChannelBuffersFromAllocs<PACKET_HEADER_TYPE, channel_allocs, SENDER_TO_ENTRY_IDX>,
        tt::tt_fabric::SenderChannelBuffersFromAllocs<PACKET_HEADER_TYPE, channel_allocs, SENDER_TO_ENTRY_IDX>>;
    auto local_sender_channels = SenderChannelsAllocT::make();

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
    auto local_sender_channel_worker_interfaces =
        tt::tt_fabric::EdmChannelWorkerInterfaces<tt::tt_fabric::worker_handshake_noc, SENDER_NUM_BUFFERS_ARRAY>::make(
            std::make_index_sequence<NUM_SENDER_CHANNELS>{});

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::DOWNSTREAM_EDM_SETUP_STARTED);
#endif

    std::array<RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC0>, VC0_DOWNSTREAM_EDM_SIZE>
        downstream_edm_noc_interfaces_vc0;

    static_assert(VC0_DOWNSTREAM_EDM_SIZE > 0, "VC0_DOWNSTREAM_EDM_SIZE must be at least 1");
#if defined(FABRIC_2D)
    [[maybe_unused]] constexpr bool use_vc0_single_live_stateful_noc =
        use_2d_single_live_stateful_noc<VC0_DOWNSTREAM_EDM_SIZE>();
#else
    [[maybe_unused]] constexpr bool use_vc0_single_live_stateful_noc = false;
#endif

    populate_local_sender_channel_free_slots_stream_id_ordered_map(
        has_downstream_edm_vc0_buffer_connection, local_sender_channel_free_slots_stream_ids);

#if !defined(FABRIC_DRISC_OFFLOAD)
    if (has_downstream_edm_vc0_buffer_connection) {
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
                        downstream_edm_vc0_worker_registration_ids[dense_index],
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
                        receiver_channel_free_slots_stream_id,
                        receiver_channel_forwarding_data_cmd_buf_ids[0],
                        receiver_channel_forwarding_sync_cmd_buf_ids[0]);
                if constexpr (is_2d_fabric) {
                    if constexpr (use_vc0_single_live_stateful_noc) {
                        downstream_edm_noc_interfaces_vc0[compact_index]
                            .template setup_edm_noc_cmd_buf<
                                tt::tt_fabric::edm_to_downstream_noc,
                                tt::tt_fabric::forward_and_local_write_noc_vc>();
                    }
                } else {
                    if constexpr (NUM_ACTIVE_ERISCS == 1 && !FORCE_ALL_PATHS_TO_USE_SAME_NOC) {
                        downstream_edm_noc_interfaces_vc0[compact_index]
                            .template setup_edm_noc_cmd_buf<
                                tt::tt_fabric::edm_to_downstream_noc,
                                tt::tt_fabric::forward_and_local_write_noc_vc>();
                    }
                }
                dense_index++;
            }
            compact_index++;
            has_downstream_edm >>= 1;
        }
    }
#endif  // !FABRIC_DRISC_OFFLOAD

    std::array<RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC1>, VC1_DOWNSTREAM_EDM_SIZE>
        downstream_edm_noc_interfaces_vc1;

#if defined(FABRIC_2D_VC1_ACTIVE)
    static_assert(VC1_DOWNSTREAM_EDM_SIZE > 0, "VC1_DOWNSTREAM_EDM_SIZE must be at least 1");
    if (has_downstream_edm_vc1_buffer_connection) {
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
                StreamId{local_tensix_relay_free_slots_stream_id},
                StreamId{tensix_relay_local_free_slots_stream_id},
                receiver_channel_forwarding_data_cmd_buf_ids[0],
                receiver_channel_forwarding_sync_cmd_buf_ids[0]);

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

    local_receiver_channels.init<channel_allocs>(channel_buffer_size, sizeof(PACKET_HEADER_TYPE));
    remote_receiver_channels.init<eth_remote_channel_allocs>(channel_buffer_size, sizeof(PACKET_HEADER_TYPE));
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
    if constexpr (num_ds_or_local_tensix_connections) {
        wait_for_notification<ENABLE_RISC_CPU_DATA_CACHE>(
            (uint32_t)edm_local_tensix_sync_ptr_addr, num_ds_or_local_tensix_connections, termination_signal_ptr);
    }

#if defined(FABRIC_DRISC_OFFLOAD)
    if constexpr (is_receiver_channel_serviced[0]) {
        constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
        noc_async_write_one_packet(
            ROUTING_TABLE_BASE,
            get_noc_addr(exp_push_drisc_x, exp_push_drisc_y, drisc_routing_table_l1_addr, noc),
            sizeof(tt::tt_fabric::routing_l1_info_t),
            noc);
        noc_async_write_barrier(noc);
        noc_async_write_one_packet(
            reinterpret_cast<uint32_t>(edm_status_ptr),
            get_noc_addr(exp_push_drisc_x, exp_push_drisc_y, drisc_e1_ready_l1_addr, noc),
            sizeof(uint32_t),
            noc);
        noc_async_write_barrier(noc);
        while (get_ptr_val<drisc_free_slots_stream_id>() == 0 &&
               !got_immediate_termination_signal<ENABLE_RISC_CPU_DATA_CACHE>(termination_signal_ptr)) {
        }
        drisc_offload_next_slot_addr = drisc_ring_base_addr;
        drisc_offload_init_noc_state();
    }
    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }
#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::VCS_OPENED);
#endif
#else
    if constexpr (is_2d_fabric) {
        auto open_downstream_edm_connections =
            [](auto& downstream_edm_noc_interfaces, uint32_t has_downstream_edm, int receiver_channel_idx) {
                uint32_t edm_index = 0;
                if (is_receiver_channel_serviced[receiver_channel_idx]) {
                    while (has_downstream_edm) {
                        if (has_downstream_edm & 0x1) {
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
#endif  // FABRIC_DRISC_OFFLOAD
    std::array<uint8_t, num_eth_ports> port_direction_table;

    if constexpr (NUM_ACTIVE_ERISCS > 1) {
        wait_for_other_local_erisc();
    }

#if !defined(FABRIC_2D_VC1_ACTIVE)
    POSTCODE(tt::tt_fabric::EDMStatus::ROUTING_TABLE_INITIALIZED);
#endif

    WAYPOINT("FSCW");
    wait_for_static_connection_to_ready(
        local_sender_channel_worker_interfaces, local_sender_channel_free_slots_stream_ids, termination_signal_ptr);
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
