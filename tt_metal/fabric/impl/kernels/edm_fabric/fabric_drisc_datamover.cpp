// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "experimental/drisc_mode.h"

#include "fabric/fabric_edm_packet_header.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"
#include "hostdevcommon/fabric_common.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_adapter.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_header_validate.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_transaction_id_tracker.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/router_data_cache.hpp"

#ifdef FABRIC_2D
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

using namespace tt::tt_fabric;
using LowLatencyFields = tt::tt_fabric::RoutingFieldsConstants::LowLatency;

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
#endif

static constexpr std::array<uint32_t, NUM_ROUTER_CARDINAL_DIRECTIONS> vc_0_free_slots_stream_ids = {
    vc_0_free_slots_from_downstream_edge_1_stream_id,
    vc_0_free_slots_from_downstream_edge_2_stream_id,
    vc_0_free_slots_from_downstream_edge_3_stream_id,
    vc_0_free_slots_from_downstream_edge_4_stream_id};

template <size_t Compact>
constexpr uint32_t vc0_local_free_slots_stream_id() {
    return vc_0_free_slots_stream_ids[Compact];
}
constexpr uint32_t vc0_local_free_slots_stream_id(size_t compact) { return vc_0_free_slots_stream_ids[compact]; }

static_assert(
    vc_0_free_slots_from_downstream_edge_1_stream_id != drisc_doorbell_stream_id &&
        vc_0_free_slots_from_downstream_edge_2_stream_id != drisc_doorbell_stream_id &&
        vc_0_free_slots_from_downstream_edge_3_stream_id != drisc_doorbell_stream_id &&
        vc_0_free_slots_from_downstream_edge_4_stream_id != drisc_doorbell_stream_id,
    "DRISC doorbell stream register collides with a downstream free-slots stream register");
static_assert(
    drisc_doorbell_stream_id < DRAM_NOC_NUM_STREAMS &&
        vc_0_free_slots_from_downstream_edge_1_stream_id < DRAM_NOC_NUM_STREAMS &&
        vc_0_free_slots_from_downstream_edge_2_stream_id < DRAM_NOC_NUM_STREAMS &&
        vc_0_free_slots_from_downstream_edge_3_stream_id < DRAM_NOC_NUM_STREAMS &&
        vc_0_free_slots_from_downstream_edge_4_stream_id < DRAM_NOC_NUM_STREAMS,
    "DRISC stream register beyond the DRAM tile's stream registers");

static_assert(
    tt::tt_fabric::edm_to_downstream_noc == 0 && tt::tt_fabric::edm_to_local_chip_noc == 1,
    "DRISC offload expects downstream forwarding on NoC0 and local delivery on NoC1");

constexpr uint8_t drisc_credit_return_cmd_buf = write_reg_cmd_buf;
static_assert(
    drisc_credit_return_cmd_buf != tt::tt_fabric::local_chip_data_cmd_buf &&
        drisc_credit_return_cmd_buf != write_at_cmd_buf,
    "DRISC credit-return command buffer on NoC1 must not collide with local data or atomic command buffers");

FORCE_INLINE void drisc_credit_doorbell_set_state() {
    constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
    noc_inline_dw_write_set_state</*posted=*/true, /*set_val=*/true>(
        get_noc_addr(
            peer_erisc_noc_x,
            peer_erisc_noc_y,
            STREAM_REG_ADDR(drisc_free_slots_stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX),
            noc),
        1u << REMOTE_DEST_BUF_WORDS_FREE_INC,
        0xF,
        drisc_credit_return_cmd_buf,
        noc,
        tt::tt_fabric::forward_and_local_write_noc_vc);
}

FORCE_INLINE void drisc_send_credits_to_e1(uint32_t count) {
    constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
    noc_inline_dw_write_with_state<
        /*update_addr_lo=*/false,
        /*update_counter=*/true,
        /*posted=*/true,
        /*update_addr_hi=*/false,
        /*update_val=*/true,
        InlineWriteDst::REG>(count << REMOTE_DEST_BUF_WORDS_FREE_INC, 0, drisc_credit_return_cmd_buf, noc);
}

// TT_FABRIC_PHASE_PROFILE: this DRISC's phase profile (fabric_phase_profile_types.hpp); compiles away when disabled.
// flush() stages the block in DRISC L1, and drisc_phase_profile_publish copies it into block PHASE_PROFILE_DRISC_BLOCK
// of E1's profile buffer, where the host reads it.
using DriscPhaseProfiler = FabricPhaseProfiler<
    drisc_phase_profile_mode,
    drisc_phase_profile_staging_addr,
    0,
    phase_profile_mask,
    phase_profile_iter_timing,
    0>;
static_assert(
    drisc_phase_profile_mode == 0 || drisc_phase_profile_staging_addr + sizeof(FabricPhaseProfile) <=
                                         exp_push_drisc_l1_base + exp_push_drisc_l1_size,
    "DRISC phase profile staging block exceeds DRISC UNRESERVED L1");
// The copy goes out on NoC1 through read_cmd_buf, which nothing else on the DRISC uses on NoC1.
static_assert(
    read_cmd_buf != tt::tt_fabric::local_chip_data_cmd_buf && read_cmd_buf != write_at_cmd_buf &&
        read_cmd_buf != drisc_credit_return_cmd_buf,
    "DRISC phase profile copy needs a NoC1 command buffer of its own");
// A global, as on the router, so it counts against the DRISC's 8 KB local data memory at link time.
DriscPhaseProfiler phase_profiler;

FORCE_INLINE void drisc_phase_profile_publish() {
    if constexpr (drisc_phase_profile_mode != 0) {
        constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
        // The previous copy must have left the staging block before flush() rewrites it.
        noc_async_posted_writes_flushed(noc);
        phase_profiler.flush();
        noc_async_write_one_packet_with_trid</*update_counter=*/true, /*posted=*/true>(
            drisc_phase_profile_staging_addr,
            get_noc_addr(
                peer_erisc_noc_x,
                peer_erisc_noc_y,
                phase_profile_buffer_addr + PHASE_PROFILE_DRISC_BLOCK * sizeof(FabricPhaseProfile),
                noc),
            sizeof(FabricPhaseProfile),
            0,
            read_cmd_buf,
            noc,
            tt::tt_fabric::forward_and_local_write_noc_vc);
    }
}

FORCE_INLINE bool local_delivery_uses_slot_trid(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_header, tt::tt_fabric::NocSendType noc_send_type) {
    if (noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE) [[unlikely]] {
        const auto& sparse = packet_header->command_fields.sparse_mcast_write;
        return sparse.counts[sparse.chip_idx] != 0;
    }
    return noc_send_type != tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC &&
           noc_send_type != tt::tt_fabric::NocSendType::NOC_UNICAST_INLINE_WRITE;
}

template <typename DownstreamSenderT>
FORCE_INLINE bool can_forward_packet_completely(
    ROUTING_FIELDS_TYPE cached_routing_fields, [[maybe_unused]] DownstreamSenderT& downstream_edm_interface) {
    bool deliver_locally_only;
    if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::RoutingFields>) {
        deliver_locally_only = cached_routing_fields.value == tt::tt_fabric::RoutingFields::LAST_MCAST_VAL;
    } else if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::LowLatencyRoutingFields>) {
        deliver_locally_only =
            (cached_routing_fields.value & LowLatencyFields::FIELD_MASK) == LowLatencyFields::WRITE_ONLY;
    }
    return deliver_locally_only || (get_ptr_val<vc0_local_free_slots_stream_id<0>()>() != 0);
}

constexpr size_t direction_to_compact_index_map[eth_chan_directions::COUNT][eth_chan_directions::COUNT] = {
    {0, 0, 1, 2, 3},  // EAST router -> WEST, NORTH, SOUTH, Z
    {0, 0, 1, 2, 3},  // WEST router -> EAST, NORTH, SOUTH, Z
    {0, 1, 0, 2, 3},  // NORTH router -> EAST, WEST, SOUTH, Z
    {0, 1, 2, 0, 3},  // SOUTH router -> EAST, WEST, NORTH, Z
    {0, 1, 2, 3, 0},  // Z router -> EAST, WEST, NORTH, SOUTH
};

template <eth_chan_directions downstream_direction>
FORCE_INLINE constexpr size_t get_downstream_edm_interface_index() {
    return direction_to_compact_index_map[my_direction][downstream_direction];
}

FORCE_INLINE constexpr size_t get_downstream_edm_interface_index(eth_chan_directions downstream_direction) {
    return direction_to_compact_index_map[my_direction][downstream_direction];
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
        return get_ptr_val<vc0_local_free_slots_stream_id<edm_index>()>() != 0;
    }
}

#ifndef FABRIC_2D
template <uint8_t rx_channel_id, typename DownstreamSenderT>
FORCE_INLINE void receiver_forward_packet(
    tt_l1_ptr PACKET_HEADER_TYPE* packet_start,
    ROUTING_FIELDS_TYPE cached_routing_fields,
    DownstreamSenderT& downstream_edm_interface,
    uint8_t transaction_id,
    bool& used_noc0,
    bool& used_noc1) {
    constexpr bool ENABLE_STATEFUL_NOC_APIS =
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
        true;
#else
        false;
#endif
    router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
    const auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_start);
    const uint16_t payload_size_bytes = packed.payload_size_bytes;
    if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::RoutingFields>) {
        bool start_distance_is_terminal_value =
            (cached_routing_fields.value & tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK) ==
            tt::tt_fabric::RoutingFields::LAST_HOP_DISTANCE_VAL;
        bool not_last_destination_device = cached_routing_fields.value != tt::tt_fabric::RoutingFields::LAST_MCAST_VAL;
        if (not_last_destination_device) {
            forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                packet_start, payload_size_bytes, cached_routing_fields, downstream_edm_interface, transaction_id);
            used_noc0 = true;
        }
        if (start_distance_is_terminal_value) {
            execute_chip_unicast_to_local_chip_impl(
                packet_start, payload_size_bytes, packed.noc_send_type, transaction_id, rx_channel_id);
            used_noc1 = local_delivery_uses_slot_trid(packet_start, packed.noc_send_type);
        }
    } else if constexpr (std::is_same_v<ROUTING_FIELDS_TYPE, tt::tt_fabric::LowLatencyRoutingFields>) {
        const uint32_t routing = cached_routing_fields.value & LowLatencyFields::FIELD_MASK;
        switch (routing) {
            case LowLatencyFields::WRITE_ONLY:
                execute_chip_unicast_to_local_chip_impl(
                    packet_start, payload_size_bytes, packed.noc_send_type, transaction_id, rx_channel_id);
                used_noc1 = local_delivery_uses_slot_trid(packet_start, packed.noc_send_type);
                break;
            case LowLatencyFields::FORWARD_ONLY: {
                forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                    packet_start, payload_size_bytes, cached_routing_fields, downstream_edm_interface, transaction_id);
                used_noc0 = true;
            } break;
            case LowLatencyFields::WRITE_AND_FORWARD: {
                if (packed.noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE) [[unlikely]] {
                    used_noc1 = local_delivery_uses_slot_trid(packet_start, packed.noc_send_type);
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
                    used_noc0 = true;
                } else {
                    forward_payload_to_downstream_edm<enable_deadlock_avoidance, ENABLE_STATEFUL_NOC_APIS>(
                        packet_start,
                        payload_size_bytes,
                        cached_routing_fields,
                        downstream_edm_interface,
                        transaction_id);
                    used_noc0 = true;
                    execute_chip_unicast_to_local_chip_impl(
                        packet_start, payload_size_bytes, packed.noc_send_type, transaction_id, rx_channel_id);
                    used_noc1 = local_delivery_uses_slot_trid(packet_start, packed.noc_send_type);
                }
            } break;
            default: {
                ASSERT(false);
            }
        }
    }
}
#endif  // !FABRIC_2D

#if defined(FABRIC_2D)
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

template <size_t DOWNSTREAM_EDM_SIZE>
constexpr bool use_2d_single_live_stateful_noc() {
    constexpr auto my_dir = static_cast<eth_chan_directions>(my_direction);
    constexpr bool is_ew = my_dir == eth_chan_directions::EAST || my_dir == eth_chan_directions::WEST;
    constexpr uint8_t LIVE = live_eth_mask_2d<DOWNSTREAM_EDM_SIZE>();
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED) && !defined(FABRIC_2D_VC1_ACTIVE)
    return is_ew && LIVE == 0b0001 && !udm_mode && !is_intermesh_router && !is_intermesh_router_on_edge &&
           !is_intramesh_router_on_edge;
#else
    return false;
#endif
}

template <typename DownstreamSenderT, size_t DOWNSTREAM_EDM_SIZE>
FORCE_INLINE __attribute__((optimize("no-jump-tables"))) bool admit_2d_dispatch(
    uint8_t action, [[maybe_unused]] std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces) {
    constexpr uint8_t LIVE = live_eth_mask_2d<DOWNSTREAM_EDM_SIZE>();
    constexpr auto dirs = Routing2DCodec::fwd_dirs<static_cast<eth_chan_directions>(my_direction)>();
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
        // Hoist stream-register reads back-to-back so the MMIO loads pipeline on multi-direction routers.
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

template <typename DownstreamSenderT, size_t DOWNSTREAM_EDM_SIZE>
FORCE_INLINE __attribute__((optimize("no-jump-tables"))) bool forward_2d_dispatch_first(
    uint8_t& action,
    tt_l1_ptr PACKET_HEADER_TYPE* packet_start,
    uint16_t payload_size_bytes,
    ROUTING_FIELDS_TYPE cached_routing_fields,
    std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    uint8_t transaction_id) {
    constexpr uint8_t LIVE = live_eth_mask_2d<DOWNSTREAM_EDM_SIZE>();
    constexpr auto dirs = Routing2DCodec::fwd_dirs<static_cast<eth_chan_directions>(my_direction)>();
    constexpr bool stateful_api = use_2d_single_live_stateful_noc<DOWNSTREAM_EDM_SIZE>();
    if constexpr (LIVE == 0b0001) {
        if (action & Routing2DCodec::action_bit(dirs[0])) {
            constexpr auto edm_index = get_downstream_edm_interface_index<dirs[0]>();
            forward_payload_to_downstream_edm<enable_deadlock_avoidance, stateful_api>(
                packet_start,
                payload_size_bytes,
                cached_routing_fields,
                downstream_edm_interfaces[edm_index],
                transaction_id);
            return true;
        }
        return false;
    } else {
        if ((action & Routing2DCodec::ACTION_ETH_MASK) == 0) {
            return false;
        }
        if constexpr ((LIVE >> 0) & 1) {
            if (action & Routing2DCodec::action_bit(dirs[0])) {
                action &= ~Routing2DCodec::action_bit(dirs[0]);
                constexpr auto edm_index = get_downstream_edm_interface_index<dirs[0]>();
                forward_payload_to_downstream_edm<
                    enable_deadlock_avoidance,
                    stateful_api,
                    /*increment_pointers=*/true,
                    /*update_local_free_slots=*/false>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces[edm_index],
                    transaction_id);
                increment_local_update_ptr_val<vc0_local_free_slots_stream_id<edm_index>()>(-1);
                return true;
            }
        }
        if constexpr ((LIVE >> 1) & 1) {
            if (action & Routing2DCodec::action_bit(dirs[1])) {
                action &= ~Routing2DCodec::action_bit(dirs[1]);
                constexpr auto edm_index = get_downstream_edm_interface_index<dirs[1]>();
                forward_payload_to_downstream_edm<
                    enable_deadlock_avoidance,
                    stateful_api,
                    /*increment_pointers=*/true,
                    /*update_local_free_slots=*/false>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces[edm_index],
                    transaction_id);
                increment_local_update_ptr_val<vc0_local_free_slots_stream_id<edm_index>()>(-1);
                return true;
            }
        }
        if constexpr ((LIVE >> 2) & 1) {
            if (action & Routing2DCodec::action_bit(dirs[2])) {
                action &= ~Routing2DCodec::action_bit(dirs[2]);
                constexpr auto edm_index = get_downstream_edm_interface_index<dirs[2]>();
                forward_payload_to_downstream_edm<
                    enable_deadlock_avoidance,
                    stateful_api,
                    /*increment_pointers=*/true,
                    /*update_local_free_slots=*/false>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces[edm_index],
                    transaction_id);
                increment_local_update_ptr_val<vc0_local_free_slots_stream_id<edm_index>()>(-1);
                return true;
            }
        }
        if constexpr ((LIVE >> 3) & 1) {
            if (action & Routing2DCodec::action_bit(dirs[3])) {
                action &= ~Routing2DCodec::action_bit(dirs[3]);
                constexpr auto edm_index = get_downstream_edm_interface_index<dirs[3]>();
                forward_payload_to_downstream_edm<
                    enable_deadlock_avoidance,
                    stateful_api,
                    /*increment_pointers=*/true,
                    /*update_local_free_slots=*/false>(
                    packet_start,
                    payload_size_bytes,
                    cached_routing_fields,
                    downstream_edm_interfaces[edm_index],
                    transaction_id);
                increment_local_update_ptr_val<vc0_local_free_slots_stream_id<edm_index>()>(-1);
                return true;
            }
        }
        return false;
    }
}

template <typename DownstreamSenderT, size_t DOWNSTREAM_EDM_SIZE>
FORCE_INLINE __attribute__((optimize("no-jump-tables"))) void forward_2d_dispatch_rest(
    uint8_t action,
    tt_l1_ptr PACKET_HEADER_TYPE* packet_start,
    uint16_t payload_size_bytes,
    ROUTING_FIELDS_TYPE cached_routing_fields,
    std::array<DownstreamSenderT, DOWNSTREAM_EDM_SIZE>& downstream_edm_interfaces,
    uint8_t transaction_id) {
    constexpr uint8_t LIVE = live_eth_mask_2d<DOWNSTREAM_EDM_SIZE>();
    constexpr auto dirs = Routing2DCodec::fwd_dirs<static_cast<eth_chan_directions>(my_direction)>();
    constexpr bool stateful_api = use_2d_single_live_stateful_noc<DOWNSTREAM_EDM_SIZE>();
    // dirs[0] was already handled and cleared by forward_2d_dispatch_first.
    if constexpr ((LIVE >> 1) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[1])) {
            constexpr auto edm_index = get_downstream_edm_interface_index<dirs[1]>();
            forward_payload_to_downstream_edm<
                enable_deadlock_avoidance,
                stateful_api,
                /*increment_pointers=*/true,
                /*update_local_free_slots=*/false>(
                packet_start,
                payload_size_bytes,
                cached_routing_fields,
                downstream_edm_interfaces[edm_index],
                transaction_id);
            increment_local_update_ptr_val<vc0_local_free_slots_stream_id<edm_index>()>(-1);
        }
    }
    if constexpr ((LIVE >> 2) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[2])) {
            constexpr auto edm_index = get_downstream_edm_interface_index<dirs[2]>();
            forward_payload_to_downstream_edm<
                enable_deadlock_avoidance,
                stateful_api,
                /*increment_pointers=*/true,
                /*update_local_free_slots=*/false>(
                packet_start,
                payload_size_bytes,
                cached_routing_fields,
                downstream_edm_interfaces[edm_index],
                transaction_id);
            increment_local_update_ptr_val<vc0_local_free_slots_stream_id<edm_index>()>(-1);
        }
    }
    if constexpr ((LIVE >> 3) & 1) {
        if (action & Routing2DCodec::action_bit(dirs[3])) {
            constexpr auto edm_index = get_downstream_edm_interface_index<dirs[3]>();
            forward_payload_to_downstream_edm<
                enable_deadlock_avoidance,
                stateful_api,
                /*increment_pointers=*/true,
                /*update_local_free_slots=*/false>(
                packet_start,
                payload_size_bytes,
                cached_routing_fields,
                downstream_edm_interfaces[edm_index],
                transaction_id);
            increment_local_update_ptr_val<vc0_local_free_slots_stream_id<edm_index>()>(-1);
        }
    }
}
#endif  // FABRIC_2D

void kernel_main() {
    ASSERT(!experimental::drisc_is_noc2axi_mode(0) && !experimental::drisc_is_noc2axi_mode(1));
    noc_local_state_init(0);
    noc_local_state_init(1);

    init_ptr_val<drisc_doorbell_stream_id>(0);

    // Skip the 2 * MAX_NUM_SENDER_CHANNELS sender semaphore args at the front of the shared runtime args.
    size_t arg_idx = 2 * MAX_NUM_SENDER_CHANNELS;
    const auto has_downstream_edm_vc0_buffer_connection = get_arg_val<uint32_t>(arg_idx++);

#if defined(FABRIC_2D)
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_buffer_base_addresses;
    for (size_t i = 0; i < NUM_DOWNSTREAM_SENDERS_VC0; i++) {
        downstream_edm_vc0_buffer_base_addresses[i] = get_arg_val<uint32_t>(arg_idx++);
    }
#else
    const auto downstream_edm_vc0_buffer_base_address = get_arg_val<uint32_t>(arg_idx++);
#endif

    const auto downstream_edm_vc0_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_edm_vc0_noc_y = get_arg_val<uint32_t>(arg_idx++);

#if defined(FABRIC_2D)
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_worker_registration_ids;
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_worker_location_info_addresses;
    std::array<uint32_t, NUM_DOWNSTREAM_SENDERS_VC0> downstream_edm_vc0_buffer_index_semaphore_addresses;
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
    const auto downstream_edm_vc0_worker_registration_id = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_edm_vc0_worker_location_info_address = get_arg_val<uint32_t>(arg_idx++);
    const auto downstream_edm_vc0_buffer_index_semaphore_address = get_arg_val<uint32_t>(arg_idx++);
#endif

    std::array<RouterToRouterSender<DOWNSTREAM_SENDER_NUM_BUFFERS_VC0>, VC0_DOWNSTREAM_EDM_SIZE>
        downstream_edm_noc_interfaces_vc0;

    if (has_downstream_edm_vc0_buffer_connection) {
        uint32_t has_downstream_edm = has_downstream_edm_vc0_buffer_connection & 0xF;
        uint32_t compact_index = 0;
        uint32_t dense_index = 0;
        while (has_downstream_edm) {
            if (has_downstream_edm & 0x1) {
                const uint32_t teardown_sem_address = drisc_teardown_sem_base_addr + dense_index * 16;
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
#if defined(FABRIC_2D)
                        downstream_edm_vc0_buffer_base_addresses[dense_index],
#else
                        downstream_edm_vc0_buffer_base_address,
#endif
                        DOWNSTREAM_SENDER_NUM_BUFFERS_VC0,
#if defined(FABRIC_2D)
                        downstream_edm_vc0_worker_registration_ids[dense_index],
                        downstream_edm_vc0_worker_location_info_addresses[dense_index],
#else
                        downstream_edm_vc0_worker_registration_id,
                        downstream_edm_vc0_worker_location_info_address,
#endif
                        channel_buffer_size,
#if defined(FABRIC_2D)
                        downstream_edm_vc0_buffer_index_semaphore_addresses[dense_index],
#else
                        downstream_edm_vc0_buffer_index_semaphore_address,
#endif
                        0,
                        reinterpret_cast<volatile uint32_t* const>(teardown_sem_address),
                        0,
#if defined(FABRIC_2D)
                        get_vc0_downstream_sender_channel_free_slots_stream_id(compact_index),
#else
                        sender_channel_1_free_slots_stream_id,
#endif
                        receiver_channel_free_slots_stream_id,
                        receiver_channel_forwarding_data_cmd_buf_ids[0],
                        receiver_channel_forwarding_sync_cmd_buf_ids[0]);
                dense_index++;
            }
            compact_index++;
            has_downstream_edm >>= 1;
        }
    }

    WriteTransactionIdTracker<
        drisc_ring_slots,
        16,
        0,
        tt::tt_fabric::edm_to_local_chip_noc,
        tt::tt_fabric::edm_to_downstream_noc>
        drisc_trid_tracker;
    drisc_trid_tracker.init();

    volatile tt_l1_ptr uint32_t* const drisc_e1_ready_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(drisc_e1_ready_l1_addr);
    volatile tt_l1_ptr uint32_t* const drisc_termination_signal_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(drisc_termination_signal_l1_addr);

    // Wait for E1 to finish the Ethernet handshake (after all local E0s have zeroed their connection semaphores)
    // and copy routing_l1_info_t into our L1.
    while (*drisc_e1_ready_ptr == 0 &&
           *drisc_termination_signal_ptr != tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE) {
        invalidate_l1_cache();
    }

    if (*drisc_termination_signal_ptr != tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE) {
        constexpr bool use_posted_writes_for_connection_open = false;
        if constexpr (is_2d_fabric) {
            uint32_t has_downstream_edm = has_downstream_edm_vc0_buffer_connection & 0xF;
            uint32_t edm_index = 0;
            while (has_downstream_edm) {
                if (has_downstream_edm & 0x1) {
                    downstream_edm_noc_interfaces_vc0[edm_index]
                        .template open<
                            false,
                            use_posted_writes_for_connection_open,
                            tt::tt_fabric::worker_handshake_noc>();
                }
                edm_index++;
                has_downstream_edm >>= 1;
            }
        } else {
            if (has_downstream_edm_vc0_buffer_connection) {
                downstream_edm_noc_interfaces_vc0[0]
                    .template open<false, use_posted_writes_for_connection_open, tt::tt_fabric::worker_handshake_noc>();
            }
        }

        {
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
            if constexpr (use_2d_single_live_stateful_noc<VC0_DOWNSTREAM_EDM_SIZE>()) {
                if (has_downstream_edm_vc0_buffer_connection & 0x1) {
                    downstream_edm_noc_interfaces_vc0[0]
                        .template setup_edm_noc_cmd_buf<
                            tt::tt_fabric::edm_to_downstream_noc,
                            tt::tt_fabric::forward_and_local_write_noc_vc>();
                }
            }
#endif
        }

        // Set up NoC1 credit-return command buffer and release E1 from its init barrier with all 16 ring credits.
#if defined(ARCH_BLACKHOLE) && !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
        drisc_local_write_init_cmd_buf();
#endif
        drisc_credit_doorbell_set_state();
        drisc_send_credits_to_e1(drisc_ring_slots);
        noc_async_posted_writes_flushed(tt::tt_fabric::edm_to_local_chip_noc);

        [[maybe_unused]] const auto& routing_table =
            *reinterpret_cast<const tt_l1_ptr tt::tt_fabric::routing_l1_info_t*>(drisc_routing_table_l1_addr);
        [[maybe_unused]] const uint16_t my_mesh_id = routing_table.my_mesh_id;
        [[maybe_unused]] const uint8_t my_mesh_coord_y = routing_table.my_mesh_coord_y;
        [[maybe_unused]] const uint8_t my_mesh_coord_x = routing_table.my_mesh_coord_x;

        ChannelCounter<drisc_ring_slots> wr_sent_counter;
        wr_sent_counter.reset();
        uint32_t rd_slot_addr = drisc_ring_base_addr;

        // Two-stage register-resident batch job tracker. Within each NoC, all ring-sourced writes use a single
        // command buffer (cmd_buf 2 on NoC0, cmd_buf 0 on NoC1) and therefore drain in strict FIFO order. Tracking
        // the last TRID issued on NoC0 and the last TRID issued on NoC1 in a batch lets a 1..4-slot batch retire
        // with at most one register read per active NoC (0 for standalone atomics, 1 for pure forward or pure local,
        // 2 for multicast/mixed).
        constexpr uint8_t kNoTrid = 0xFF;
        uint8_t cur_count = 0;
        uint8_t cur_noc0_trid = kNoTrid;
        uint8_t cur_noc1_trid = kNoTrid;
        uint8_t pend_count = 0;
        uint8_t pend_noc0_trid = kNoTrid;
        uint8_t pend_noc1_trid = kNoTrid;

        phase_profiler.init<0, 1>();
        drisc_phase_profile_publish();
        [[maybe_unused]] uint32_t phase_profile_iters = 0;

        while (true) {
            bool progress = false;
            const int32_t pkts_available = get_ptr_val<drisc_doorbell_stream_id>();
            if (pkts_available != 0) {
                const uint32_t visit_start = phase_profiler.open<PHASE_RX_VISIT>();
                const uint32_t admit_start = phase_profiler.open<PHASE_RX_DECODE_ADMIT>();
                invalidate_l1_cache();
                const auto slot_index = wr_sent_counter.get_buffer_index();
                tt_l1_ptr PACKET_HEADER_TYPE* const packet_header =
                    reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(rd_slot_addr);

                ROUTING_FIELDS_TYPE cached_routing_fields{};
#ifndef FABRIC_2D
                cached_routing_fields = packet_header->routing_fields;
#endif
                uint8_t action = 0;
                bool local_deliver = false;
                bool intermesh_egress = false;
                uint8_t intermesh_egress_index = 0;
                bool can_send_to_all_receivers = false;

                if constexpr (is_2d_fabric) {
#if defined(FABRIC_2D)
                    if constexpr (receiver_channel_is_intermesh_ingress[0]) {
                        fabric_set_2d_intermesh_landing_route(packet_header, routing_table, MESH_Y_SIZE, MESH_X_SIZE);
                        action = Routing2DCodec::decode_action_y_first(
                            packet_header->route_buffer, my_mesh_coord_y, my_mesh_coord_x, MESH_Y_SIZE);
                    } else {
                        action = Routing2DCodec::decode_action<static_cast<eth_chan_directions>(my_direction)>(
                            packet_header->route_buffer, my_mesh_coord_y, my_mesh_coord_x, MESH_Y_SIZE);
                    }
                    bool intermesh_exit = false;
                    if constexpr (is_intramesh_router_on_edge) {
                        intermesh_exit = Routing2DCodec::action_is_intermesh_exit(
                            action, packet_header->dst_start_mesh_id, my_mesh_id);
                    }
                    if (intermesh_exit) {
                        const auto boundary_dir = static_cast<eth_chan_directions>(
                            routing_table.inter_mesh_direction_table.get_original_direction(
                                packet_header->dst_start_mesh_id));
                        const bool boundary_dir_is_addressable =
                            boundary_dir < eth_chan_directions::COUNT &&
                            boundary_dir != static_cast<eth_chan_directions>(my_direction) &&
                            get_downstream_edm_interface_index(boundary_dir) < downstream_edm_noc_interfaces_vc0.size();
                        if (!boundary_dir_is_addressable) {
                            ASSERT(false);
                            can_send_to_all_receivers = false;
                        } else {
                            intermesh_egress = true;
                            intermesh_egress_index =
                                static_cast<uint8_t>(get_downstream_edm_interface_index(boundary_dir));
                            can_send_to_all_receivers =
                                downstream_edm_noc_interfaces_vc0[intermesh_egress_index]
                                    .template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>();
                        }
                    } else {
                        local_deliver = (action & Routing2DCodec::ACTION_LOCAL_DELIVER) != 0;
                        can_send_to_all_receivers = admit_2d_dispatch(action, downstream_edm_noc_interfaces_vc0);
                    }
#endif
                } else {
#ifndef FABRIC_2D
                    can_send_to_all_receivers =
                        can_forward_packet_completely(cached_routing_fields, downstream_edm_noc_interfaces_vc0[0]);
#endif
                }
                phase_profiler.close<PHASE_RX_DECODE_ADMIT>(admit_start);

                if (can_send_to_all_receivers) {
                    progress = true;
                    bool used_noc0 = false;
                    bool used_noc1 = false;
                    const uint8_t trid =
                        drisc_trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(slot_index);
                    if constexpr (!is_receiver_channel_forwarding_disabled[0]) {
                        if constexpr (is_2d_fabric) {
#if defined(FABRIC_2D)
                            const auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_header);
                            const uint32_t forward_start = phase_profiler.open<PHASE_RX_FORWARD>();
                            if (intermesh_egress) {
                                forward_payload_to_downstream_edm<enable_deadlock_avoidance, false>(
                                    packet_header,
                                    packed.payload_size_bytes,
                                    cached_routing_fields,
                                    downstream_edm_noc_interfaces_vc0[intermesh_egress_index],
                                    trid);
                                used_noc0 = true;
                                phase_profiler.close<PHASE_RX_FORWARD>(forward_start);
                            } else {
                                used_noc0 = forward_2d_dispatch_first(
                                    action,
                                    packet_header,
                                    packed.payload_size_bytes,
                                    cached_routing_fields,
                                    downstream_edm_noc_interfaces_vc0,
                                    trid);
                                // Counted only when the packet was forwarded; a local-only packet's dispatch checks
                                // fall into the visit's remainder.
                                if (used_noc0) {
                                    phase_profiler.close<PHASE_RX_FORWARD>(forward_start);
                                }
                                if (local_deliver) {
                                    const uint32_t local_start = phase_profiler.open<PHASE_RX_LOCAL>();
                                    execute_chip_unicast_to_local_chip_impl(
                                        packet_header, packed.payload_size_bytes, packed.noc_send_type, trid, 0);
                                    used_noc1 = local_delivery_uses_slot_trid(packet_header, packed.noc_send_type);
                                    phase_profiler.close<PHASE_RX_LOCAL>(local_start);
                                }
                                if constexpr (live_eth_mask_2d<VC0_DOWNSTREAM_EDM_SIZE>() != 0b0001) {
                                    if ((action & Routing2DCodec::ACTION_ETH_MASK) != 0) {
                                        const uint32_t forward_rest_start = phase_profiler.open<PHASE_RX_FORWARD>();
                                        forward_2d_dispatch_rest(
                                            action,
                                            packet_header,
                                            packed.payload_size_bytes,
                                            cached_routing_fields,
                                            downstream_edm_noc_interfaces_vc0,
                                            trid);
                                        phase_profiler.close<PHASE_RX_FORWARD>(forward_rest_start);
                                    }
                                }
                            }
#endif
                        } else {
#ifndef FABRIC_2D
                            receiver_forward_packet<0>(
                                packet_header,
                                cached_routing_fields,
                                downstream_edm_noc_interfaces_vc0[0],
                                trid,
                                used_noc0,
                                used_noc1);
#endif
                        }
                    }
                    if (used_noc0) {
                        cur_noc0_trid = trid;
                    }
                    if (used_noc1) {
                        cur_noc1_trid = trid;
                    }
                    cur_count++;
                    wr_sent_counter.increment();
                    const uint32_t next_slot_addr = rd_slot_addr + channel_buffer_size;
                    rd_slot_addr = next_slot_addr == drisc_ring_end_addr ? drisc_ring_base_addr : next_slot_addr;
                    increment_local_update_ptr_val<drisc_doorbell_stream_id>(-1);
                    phase_profiler.count_pkt();
                    phase_profiler.mark_progress();
                    fabric_experiment_spin<fabric_exp_drisc_delay>();
                } else {
                    phase_profiler.mark_blocked();
                    phase_profiler.count_event(EVENT_RX_BLOCKED_ADMIT);
                }
                phase_profiler.close<PHASE_RX_VISIT>(visit_start);
            }

            constexpr uint8_t batch_target = 4;
            if (pend_count != 0 && (cur_count != 0 || !progress)) {
                const uint32_t completion_start = phase_profiler.open<PHASE_RX_COMPLETION>();
                const bool noc0_done =
                    pend_noc0_trid == kNoTrid || ncrisc_noc_nonposted_write_with_transaction_id_sent(
                                                     tt::tt_fabric::edm_to_downstream_noc, pend_noc0_trid);
                const bool noc1_done =
                    pend_noc1_trid == kNoTrid || ncrisc_noc_nonposted_write_with_transaction_id_sent(
                                                     tt::tt_fabric::edm_to_local_chip_noc, pend_noc1_trid);
                if (noc0_done && noc1_done) {
                    drisc_send_credits_to_e1(pend_count);
                    pend_count = 0;
                    phase_profiler.close<PHASE_RX_COMPLETION>(completion_start);
                    phase_profiler.mark_progress();
                } else {
                    phase_profiler.mark_blocked();
                    phase_profiler.count_event(EVENT_RX_BLOCKED_TRID);
                }
            }
            if (pend_count == 0 && (cur_count >= batch_target || (cur_count != 0 && !progress))) {
                pend_count = cur_count;
                pend_noc0_trid = cur_noc0_trid;
                pend_noc1_trid = cur_noc1_trid;
                cur_count = 0;
                cur_noc0_trid = kNoTrid;
                cur_noc1_trid = kNoTrid;
                if (!progress) {
                    const uint32_t completion_start = phase_profiler.open<PHASE_RX_COMPLETION>();
                    const bool noc0_done =
                        pend_noc0_trid == kNoTrid || ncrisc_noc_nonposted_write_with_transaction_id_sent(
                                                         tt::tt_fabric::edm_to_downstream_noc, pend_noc0_trid);
                    const bool noc1_done =
                        pend_noc1_trid == kNoTrid || ncrisc_noc_nonposted_write_with_transaction_id_sent(
                                                         tt::tt_fabric::edm_to_local_chip_noc, pend_noc1_trid);
                    if (noc0_done && noc1_done) {
                        drisc_send_credits_to_e1(pend_count);
                        pend_count = 0;
                        phase_profiler.close<PHASE_RX_COMPLETION>(completion_start);
                        phase_profiler.mark_progress();
                    } else {
                        phase_profiler.mark_blocked();
                        phase_profiler.count_event(EVENT_RX_BLOCKED_TRID);
                    }
                }
            } else if (!progress && pend_count == 0 && cur_count == 0 && pkts_available == 0) {
                invalidate_l1_cache();
                if (*drisc_termination_signal_ptr == tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE &&
                    get_ptr_val<drisc_doorbell_stream_id>() == 0) {
                    break;
                }
            }

            phase_profiler.end_iteration();
            if constexpr (drisc_phase_profile_mode != 0) {
                if (++phase_profile_iters == PHASE_PROFILE_DRISC_FLUSH_ITERS) {
                    phase_profile_iters = 0;
                    drisc_phase_profile_publish();
                }
            }
        }
    }

    drisc_trid_tracker.all_buffer_slot_transactions_acked();
    for (uint8_t n = 0; n < NUM_NOCS; n++) {
        noc_async_posted_writes_flushed(n);
        noc_async_write_barrier(n);
        noc_async_read_barrier(n);
        noc_async_atomic_barrier(n);
    }

    // Signal E1 that all DRISC work and NoC transactions are complete (raising E1's credit counter to drisc_ring_slots
    // + 1).
    drisc_credit_doorbell_set_state();
    drisc_send_credits_to_e1(1);
    noc_async_posted_writes_flushed(tt::tt_fabric::edm_to_local_chip_noc);
    noc_async_write_barrier(tt::tt_fabric::edm_to_local_chip_noc);

    for (uint8_t n = 0; n < NUM_NOCS; n++) {
        noc_clear_packet_tags(n);
        noc_init_one(n, MEM_NOC_ATOMIC_RET_VAL_ADDR);
    }
}
