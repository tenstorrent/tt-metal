// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <tuple>

#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake_types.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/struct_layout.hpp"

namespace tt::tt_fabric::layout {

// ============ Sender channels ============

template <>
struct StructLayout<WorkerXY> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(WorkerXY, x),
        LAYOUT_MEMBER(WorkerXY, y),
    };
};
static_assert(check_struct_layout<WorkerXY>());

template <>
struct StructLayout<EDMChannelWorkerLocationInfo> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, worker_semaphore_address),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, align_pad_0),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, align_pad_1),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, align_pad_2),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, worker_teardown_semaphore_address),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, align_pad_3),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, align_pad_4),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, align_pad_5),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, worker_xy),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, align_pad_6),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, align_pad_7),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, align_pad_8),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, edm_read_counter),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, align_pad_9),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, align_pad_10),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, align_pad_11),
    };
};
static_assert(check_struct_layout<EDMChannelWorkerLocationInfo>());

template <>
struct StructLayout<SenderChannelProducerCursor> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(SenderChannelProducerCursor, write_counter),
        LAYOUT_MEMBER(SenderChannelProducerCursor, write_index),
        LAYOUT_MEMBER(SenderChannelProducerCursor, align_pad_0),
        LAYOUT_MEMBER(SenderChannelProducerCursor, align_pad_1),
    };
};
static_assert(check_struct_layout<SenderChannelProducerCursor>());

// ============ Handshake ============

template <>
struct StructLayout<erisc::datamover::handshake::handshake_info_t> {
    using T = erisc::datamover::handshake::handshake_info_t;
    static constexpr std::array members = {
        LAYOUT_MEMBER(T, local_value),
        LAYOUT_MEMBER(T, neighbor_mesh_id),
        LAYOUT_MEMBER(T, neighbor_device_id),
        LAYOUT_MEMBER(T, padding0),
        LAYOUT_MEMBER(T, padding),
        LAYOUT_MEMBER(T, scratch),
    };
};
static_assert(check_struct_layout<erisc::datamover::handshake::handshake_info_t>());

// ============ Routing table ============

template <>
struct StructLayout<RouterStateManager> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(RouterStateManager, state),
        LAYOUT_MEMBER(RouterStateManager, padding0),
        LAYOUT_MEMBER(RouterStateManager, command),
        LAYOUT_MEMBER(RouterStateManager, padding1),
    };
};
static_assert(check_struct_layout<RouterStateManager>());

template <>
struct StructLayout<routing_l1_info_t> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(routing_l1_info_t, state_manager),
        LAYOUT_MEMBER(routing_l1_info_t, my_mesh_id),
        LAYOUT_MEMBER(routing_l1_info_t, my_device_id),
        LAYOUT_BYTES(routing_l1_info_t, intra_mesh_direction_table),
        LAYOUT_BYTES(routing_l1_info_t, inter_mesh_direction_table),
        // Although the actual member that is used to represent the route table is determined by whether the run
        // is in 1D or 2D mode, we just use the 2d route table, which accounts for all the bytes that the union takes
        // up, and rely on decode to interpret it in the appropriate manner.
        LAYOUT_BYTES(routing_l1_info_t, route_table_2d),
        LAYOUT_BYTES(routing_l1_info_t, exit_node_table),
        LAYOUT_MEMBER(routing_l1_info_t, my_mesh_coord_y),
        LAYOUT_MEMBER(routing_l1_info_t, my_mesh_coord_x),
        LAYOUT_MEMBER(routing_l1_info_t, mesh_y_size),
        LAYOUT_MEMBER(routing_l1_info_t, mesh_x_size),
    };
};
static_assert(check_struct_layout<routing_l1_info_t>());

// ============ Fabric telemetry ============

template <>
struct StructLayout<RiscTimestampV2> {
    static constexpr std::array members = {
        // Union member that spans the whole struct
        LAYOUT_MEMBER(RiscTimestampV2, full),
    };
};
static_assert(check_struct_layout<RiscTimestampV2>());

template <>
struct StructLayout<BandwidthTelemetry> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(BandwidthTelemetry, elapsed_active_cycles),
        LAYOUT_MEMBER(BandwidthTelemetry, elapsed_cycles),
        LAYOUT_MEMBER(BandwidthTelemetry, num_words_sent),
        LAYOUT_MEMBER(BandwidthTelemetry, num_packets_sent),
    };
};
static_assert(check_struct_layout<BandwidthTelemetry>());

template <>
struct StructLayout<EriscDynamicEntry> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(EriscDynamicEntry, router_state),
        LAYOUT_PAD(EriscDynamicEntry, router_state, tx_heartbeat),
        LAYOUT_MEMBER(EriscDynamicEntry, tx_heartbeat),
        LAYOUT_MEMBER(EriscDynamicEntry, rx_heartbeat),
    };
};
static_assert(check_struct_layout<EriscDynamicEntry>());

template <>
struct StructLayout<DynamicInfo> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(DynamicInfo, tx_bandwidth),
        LAYOUT_MEMBER(DynamicInfo, rx_bandwidth),
        LAYOUT_MEMBER(DynamicInfo, erisc),
    };
};
static_assert(check_struct_layout<DynamicInfo>());

template <>
struct StructLayout<StaticInfo> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(StaticInfo, version),
        LAYOUT_MEMBER(StaticInfo, mesh_id),
        LAYOUT_MEMBER(StaticInfo, neighbor_mesh_id),
        LAYOUT_MEMBER(StaticInfo, device_id),
        LAYOUT_MEMBER(StaticInfo, neighbor_device_id),
        LAYOUT_MEMBER(StaticInfo, direction),
        LAYOUT_MEMBER(StaticInfo, supported_stats),
        LAYOUT_MEMBER(StaticInfo, fabric_config),
    };
};
static_assert(check_struct_layout<StaticInfo>());

template <>
struct StructLayout<FabricTelemetry> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(FabricTelemetry, static_info),
        LAYOUT_MEMBER(FabricTelemetry, dynamic_info),
        LAYOUT_MEMBER(FabricTelemetry, postcode),
        LAYOUT_MEMBER(FabricTelemetry, scratch),
    };
};
static_assert(check_struct_layout<FabricTelemetry>());

// ============ All described structs ============

// Every struct described at compile time. The manifest writes one type entry per element, which describes the
// struct's members.
using DescribedStructs = std::tuple<
    WorkerXY,
    EDMChannelWorkerLocationInfo,
    SenderChannelProducerCursor,
    erisc::datamover::handshake::handshake_info_t,
    RouterStateManager,
    routing_l1_info_t,
    RiscTimestampV2,
    BandwidthTelemetry,
    EriscDynamicEntry,
    DynamicInfo,
    StaticInfo,
    FabricTelemetry>;

}  // namespace tt::tt_fabric::layout
