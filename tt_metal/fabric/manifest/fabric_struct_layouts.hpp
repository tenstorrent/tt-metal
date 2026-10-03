// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <tuple>

#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake_types.hpp"
#include "tt_metal/fabric/manifest/struct_layout.hpp"

namespace tt::tt_fabric {

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

// Every struct described at compile time. The manifest writes one type entry per element, which describes the
// struct's fields.
using DescribedStructs = std::tuple<
    WorkerXY,
    EDMChannelWorkerLocationInfo,
    erisc::datamover::handshake::handshake_info_t,
    SenderChannelProducerCursor,
    RouterStateManager,
    routing_l1_info_t>;

}  // namespace tt::tt_fabric
