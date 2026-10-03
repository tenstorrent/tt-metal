// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <tuple>
#include <vector>

#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake_types.hpp"
#include "tt_metal/fabric/manifest/struct_layout.hpp"

namespace tt::tt_metal {
class Hal;
}  // namespace tt::tt_metal

namespace tt::tt_fabric {

// ============ Sender channels ============

template <>
struct StructLayout<WorkerXY> {
    static constexpr std::array fields = {
        LAYOUT_FIELD(WorkerXY, x),
        LAYOUT_FIELD(WorkerXY, y),
    };
};
static_assert(validate_struct_fields<WorkerXY>(StructLayout<WorkerXY>::fields));

template <>
struct StructLayout<EDMChannelWorkerLocationInfo> {
    static constexpr std::array fields = {
        LAYOUT_FIELD(EDMChannelWorkerLocationInfo, worker_semaphore_address),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_0),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_1),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_2),
        LAYOUT_FIELD(EDMChannelWorkerLocationInfo, worker_teardown_semaphore_address),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_3),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_4),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_5),
        LAYOUT_FIELD(EDMChannelWorkerLocationInfo, worker_xy),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_6),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_7),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_8),
        LAYOUT_FIELD(EDMChannelWorkerLocationInfo, edm_read_counter),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_9),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_10),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_11),
    };
};
static_assert(validate_struct_fields<EDMChannelWorkerLocationInfo>(StructLayout<EDMChannelWorkerLocationInfo>::fields));

template <>
struct StructLayout<SenderChannelProducerCursor> {
    static constexpr std::array fields = {
        LAYOUT_FIELD(SenderChannelProducerCursor, write_counter),
        LAYOUT_FIELD(SenderChannelProducerCursor, write_index),
        LAYOUT_PAD(SenderChannelProducerCursor, align_pad_0),
        LAYOUT_PAD(SenderChannelProducerCursor, align_pad_1),
    };
};
static_assert(validate_struct_fields<SenderChannelProducerCursor>(StructLayout<SenderChannelProducerCursor>::fields));

// ============ Handshake ============

template <>
struct StructLayout<erisc::datamover::handshake::handshake_info_t> {
    using T = erisc::datamover::handshake::handshake_info_t;
    static constexpr std::array fields = {
        LAYOUT_FIELD(T, local_value),
        LAYOUT_FIELD(T, neighbor_mesh_id),
        LAYOUT_FIELD(T, neighbor_device_id),
        LAYOUT_PAD(T, padding0),
        LAYOUT_PAD(T, padding),
        LAYOUT_FIELD(T, scratch),
    };
};
static_assert(validate_struct_fields<erisc::datamover::handshake::handshake_info_t>(
    StructLayout<erisc::datamover::handshake::handshake_info_t>::fields));

// ============ Routing table ============

template <>
struct StructLayout<RouterStateManager> {
    static constexpr std::array fields = {
        LAYOUT_FIELD(RouterStateManager, state),
        LAYOUT_PAD(RouterStateManager, padding0),
        LAYOUT_FIELD(RouterStateManager, command),
        LAYOUT_PAD(RouterStateManager, padding1),
    };
};
static_assert(validate_struct_fields<RouterStateManager>(StructLayout<RouterStateManager>::fields));

template <>
struct StructLayout<routing_l1_info_t> {
    static constexpr std::array fields = {
        LAYOUT_FIELD(routing_l1_info_t, state_manager),
        LAYOUT_FIELD(routing_l1_info_t, my_mesh_id),
        LAYOUT_FIELD(routing_l1_info_t, my_device_id),
        LAYOUT_PACKED(routing_l1_info_t, intra_mesh_direction_table, "direction_table"),
        LAYOUT_PACKED(routing_l1_info_t, inter_mesh_direction_table, "direction_table"),
        // The union of the 1D and 2D routing tables, listed by its widest member so the size and offset checks hold.
        // Decode picks the appropriate union member based on the routing mode, which it gets from is_2d_routing.
        LAYOUT_BYTES(routing_l1_info_t, route_table_2d),
        LAYOUT_BYTES(routing_l1_info_t, exit_node_table),
        LAYOUT_FIELD(routing_l1_info_t, my_mesh_coord_y),
        LAYOUT_FIELD(routing_l1_info_t, my_mesh_coord_x),
        LAYOUT_FIELD(routing_l1_info_t, mesh_y_size),
        LAYOUT_FIELD(routing_l1_info_t, mesh_x_size),
    };
};
static_assert(validate_struct_fields<routing_l1_info_t>(StructLayout<routing_l1_info_t>::fields));

// ============ Fabric telemetry ============

template <>
struct StructLayout<RiscTimestampV2> {
    static constexpr std::array fields = {
        // Union member that spans the whole struct
        LAYOUT_FIELD(RiscTimestampV2, full),
    };
};
static_assert(validate_struct_fields<RiscTimestampV2>(StructLayout<RiscTimestampV2>::fields));

template <>
struct StructLayout<BandwidthTelemetry> {
    static constexpr std::array fields = {
        LAYOUT_FIELD(BandwidthTelemetry, elapsed_active_cycles),
        LAYOUT_FIELD(BandwidthTelemetry, elapsed_cycles),
        LAYOUT_FIELD(BandwidthTelemetry, num_words_sent),
        LAYOUT_FIELD(BandwidthTelemetry, num_packets_sent),
    };
};
static_assert(validate_struct_fields<BandwidthTelemetry>(StructLayout<BandwidthTelemetry>::fields));

template <>
struct StructLayout<EriscDynamicEntry> {
    static constexpr std::array fields = {
        LAYOUT_FIELD(EriscDynamicEntry, router_state),
        LAYOUT_PAD(EriscDynamicEntry, padding0),
        LAYOUT_FIELD(EriscDynamicEntry, tx_heartbeat),
        LAYOUT_FIELD(EriscDynamicEntry, rx_heartbeat),
    };
};
static_assert(validate_struct_fields<EriscDynamicEntry>(StructLayout<EriscDynamicEntry>::fields));

template <>
struct StructLayout<DynamicInfo> {
    static constexpr std::array fields = {
        LAYOUT_FIELD(DynamicInfo, tx_bandwidth),
        LAYOUT_FIELD(DynamicInfo, rx_bandwidth),
        LAYOUT_FIELD(DynamicInfo, erisc),
    };
};
static_assert(validate_struct_fields<DynamicInfo>(StructLayout<DynamicInfo>::fields));

template <>
struct StructLayout<StaticInfo> {
    static constexpr std::array fields = {
        LAYOUT_FIELD(StaticInfo, version),
        LAYOUT_FIELD(StaticInfo, mesh_id),
        LAYOUT_FIELD(StaticInfo, neighbor_mesh_id),
        LAYOUT_FIELD(StaticInfo, device_id),
        LAYOUT_FIELD(StaticInfo, neighbor_device_id),
        LAYOUT_FIELD(StaticInfo, direction),
        LAYOUT_FIELD(StaticInfo, supported_stats),
        LAYOUT_FIELD(StaticInfo, fabric_config),
    };
};
static_assert(validate_struct_fields<StaticInfo>(StructLayout<StaticInfo>::fields));

template <>
struct StructLayout<FabricTelemetry> {
    static constexpr std::array fields = {
        LAYOUT_FIELD(FabricTelemetry, static_info),
        LAYOUT_FIELD(FabricTelemetry, dynamic_info),
        LAYOUT_FIELD(FabricTelemetry, postcode),
        LAYOUT_FIELD(FabricTelemetry, scratch),
    };
};
static_assert(validate_struct_fields<FabricTelemetry>(StructLayout<FabricTelemetry>::fields));

// ============ All described structs ============

// Every struct described at compile time. The manifest writes one type entry per element, which describes the
// struct's fields.
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

// ============ Go message ============

// The layout of go_msg_t on the active Ethernet core, built at run time from the HAL's generated accessors. The host
// cannot name the raw go_msg_t (dev_msgs.h is compiled only inside the HAL's per-arch files), so this layout is not
// in DescribedStructs, and instead a gtest checks its coverage for each arch.
std::vector<FieldLayout> go_msg_layout(const tt::tt_metal::Hal& hal);

}  // namespace tt::tt_fabric
