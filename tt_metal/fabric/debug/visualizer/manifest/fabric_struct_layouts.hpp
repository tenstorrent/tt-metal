// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "tt_metal/fabric/channel_trimming_import.hpp"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake_types.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/code_profiling_types.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/struct_layout.hpp"

namespace tt::tt_metal {
class Hal;
}  // namespace tt::tt_metal

namespace tt::tt_fabric::layout {

// ============ Sender channels ============

template <>
struct StructLayout<WorkerXY> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(WorkerXY, x),
        LAYOUT_MEMBER(WorkerXY, y),
    };
};
static_assert(validate_struct_members<WorkerXY>(StructLayout<WorkerXY>::members));

template <>
struct StructLayout<EDMChannelWorkerLocationInfo> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, worker_semaphore_address),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_0),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_1),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_2),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, worker_teardown_semaphore_address),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_3),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_4),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_5),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, worker_xy),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_6),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_7),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_8),
        LAYOUT_MEMBER(EDMChannelWorkerLocationInfo, edm_read_counter),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_9),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_10),
        LAYOUT_PAD(EDMChannelWorkerLocationInfo, align_pad_11),
    };
};
static_assert(
    validate_struct_members<EDMChannelWorkerLocationInfo>(StructLayout<EDMChannelWorkerLocationInfo>::members));

template <>
struct StructLayout<SenderChannelProducerCursor> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(SenderChannelProducerCursor, write_counter),
        LAYOUT_MEMBER(SenderChannelProducerCursor, write_index),
        LAYOUT_PAD(SenderChannelProducerCursor, align_pad_0),
        LAYOUT_PAD(SenderChannelProducerCursor, align_pad_1),
    };
};
static_assert(validate_struct_members<SenderChannelProducerCursor>(StructLayout<SenderChannelProducerCursor>::members));

// ============ Handshake ============

template <>
struct StructLayout<erisc::datamover::handshake::handshake_info_t> {
    using T = erisc::datamover::handshake::handshake_info_t;
    static constexpr std::array members = {
        LAYOUT_MEMBER(T, local_value),
        LAYOUT_MEMBER(T, neighbor_mesh_id),
        LAYOUT_MEMBER(T, neighbor_device_id),
        LAYOUT_PAD(T, padding0),
        LAYOUT_PAD(T, padding),
        LAYOUT_MEMBER(T, scratch),
    };
};
static_assert(validate_struct_members<erisc::datamover::handshake::handshake_info_t>(
    StructLayout<erisc::datamover::handshake::handshake_info_t>::members));

// ============ Routing table ============

template <>
struct StructLayout<RouterStateManager> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(RouterStateManager, state),
        LAYOUT_PAD(RouterStateManager, padding0),
        LAYOUT_MEMBER(RouterStateManager, command),
        LAYOUT_PAD(RouterStateManager, padding1),
    };
};
static_assert(validate_struct_members<RouterStateManager>(StructLayout<RouterStateManager>::members));

template <>
struct StructLayout<routing_l1_info_t> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(routing_l1_info_t, state_manager),
        LAYOUT_MEMBER(routing_l1_info_t, my_mesh_id),
        LAYOUT_MEMBER(routing_l1_info_t, my_device_id),
        LAYOUT_PACKED(routing_l1_info_t, intra_mesh_direction_table, "direction_table"),
        LAYOUT_PACKED(routing_l1_info_t, inter_mesh_direction_table, "direction_table"),
        // The union of the 1D and 2D routing tables, listed by its widest member so the size and offset checks hold.
        // Decode picks the appropriate union member based on the routing mode, which it gets from is_2d_routing.
        LAYOUT_BYTES(routing_l1_info_t, route_table_2d),
        LAYOUT_BYTES(routing_l1_info_t, exit_node_table),
        LAYOUT_MEMBER(routing_l1_info_t, my_mesh_coord_y),
        LAYOUT_MEMBER(routing_l1_info_t, my_mesh_coord_x),
        LAYOUT_MEMBER(routing_l1_info_t, mesh_y_size),
        LAYOUT_MEMBER(routing_l1_info_t, mesh_x_size),
    };
};
static_assert(validate_struct_members<routing_l1_info_t>(StructLayout<routing_l1_info_t>::members));

// ============ Fabric telemetry ============

template <>
struct StructLayout<RiscTimestampV2> {
    static constexpr std::array members = {
        // Union member that spans the whole struct
        LAYOUT_MEMBER(RiscTimestampV2, full),
    };
};
static_assert(validate_struct_members<RiscTimestampV2>(StructLayout<RiscTimestampV2>::members));

template <>
struct StructLayout<BandwidthTelemetry> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(BandwidthTelemetry, elapsed_active_cycles),
        LAYOUT_MEMBER(BandwidthTelemetry, elapsed_cycles),
        LAYOUT_MEMBER(BandwidthTelemetry, num_words_sent),
        LAYOUT_MEMBER(BandwidthTelemetry, num_packets_sent),
    };
};
static_assert(validate_struct_members<BandwidthTelemetry>(StructLayout<BandwidthTelemetry>::members));

template <>
struct StructLayout<EriscDynamicEntry> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(EriscDynamicEntry, router_state),
        LAYOUT_PAD(EriscDynamicEntry, padding0),
        LAYOUT_MEMBER(EriscDynamicEntry, tx_heartbeat),
        LAYOUT_MEMBER(EriscDynamicEntry, rx_heartbeat),
    };
};
static_assert(validate_struct_members<EriscDynamicEntry>(StructLayout<EriscDynamicEntry>::members));

template <>
struct StructLayout<DynamicInfo> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(DynamicInfo, tx_bandwidth),
        LAYOUT_MEMBER(DynamicInfo, rx_bandwidth),
        LAYOUT_MEMBER(DynamicInfo, erisc),
    };
};
static_assert(validate_struct_members<DynamicInfo>(StructLayout<DynamicInfo>::members));

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
static_assert(validate_struct_members<StaticInfo>(StructLayout<StaticInfo>::members));

template <>
struct StructLayout<FabricTelemetry> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(FabricTelemetry, static_info),
        LAYOUT_MEMBER(FabricTelemetry, dynamic_info),
        LAYOUT_MEMBER(FabricTelemetry, postcode),
        LAYOUT_MEMBER(FabricTelemetry, scratch),
    };
};
static_assert(validate_struct_members<FabricTelemetry>(StructLayout<FabricTelemetry>::members));

// ============ Diagnostics ============

template <>
struct StructLayout<CodeProfilingTimerResult> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(CodeProfilingTimerResult, total_cycles),
        LAYOUT_MEMBER(CodeProfilingTimerResult, num_instances),
    };
};
static_assert(validate_struct_members<CodeProfilingTimerResult>(StructLayout<CodeProfilingTimerResult>::members));

// The channel trimming capture, which the kernel writes and a later run imports as its trimming profile.
template <>
struct StructLayout<ChannelTrimmingOverrides> {
    using T = ChannelTrimmingOverrides;
    static constexpr std::string_view name = "FabricDatapathUsageL1Results";
    static constexpr std::array members = {
        LAYOUT_MEMBER(T, sender_channel_min_packet_size_seen_bytes_by_vc),
        LAYOUT_MEMBER(T, sender_channel_max_packet_size_seen_bytes_by_vc),
        LAYOUT_MEMBER(T, sender_channel_used_bitfield_by_vc),
        LAYOUT_MEMBER(T, sender_channel_forwarded_to_bitfield_by_vc),
        LAYOUT_MEMBER(T, receiver_channel_data_forwarded_bitfield_by_vc),
        LAYOUT_MEMBER(T, used_noc_send_type_by_vc_bitfield),
    };
};
static_assert(validate_struct_members<ChannelTrimmingOverrides>(StructLayout<ChannelTrimmingOverrides>::members));

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
    FabricTelemetry,
    CodeProfilingTimerResult,
    ChannelTrimmingOverrides>;

template <Described T>
StructType struct_type() {
    const auto& members = StructLayout<T>::members;
    return {detail::struct_name<T>(), sizeof(T), {members.begin(), members.end()}};
}

inline std::vector<StructType> described_struct_types() {
    return []<typename... Ts>(std::type_identity<std::tuple<Ts...>>) {
        return std::vector<StructType>{struct_type<Ts>()...};
    }(std::type_identity<DescribedStructs>{});
}

// ============ HAL messages ============

// The layout of go_msg_t on the active Ethernet core, built at run time from the HAL's generated accessors. The host
// cannot name the raw go_msg_t (dev_msgs.h is compiled only inside the HAL's per-arch files), so this layout is not
// in DescribedStructs, and instead a gtest (test_struct_layouts.cpp) checks its coverage for each arch.
std::vector<Member> go_msg_layout(const tt::tt_metal::Hal& hal);

// The struct name go_msg_layout describes.
inline constexpr std::string_view go_msg_name = "go_msg_t";

// The HAL messages the manifest describes on the active Ethernet core: go_msg_t, and launch_msg_t with every struct
// it holds, each member named by the HAL's generated field. Sizes, offsets and array lengths are the arch's.
std::vector<StructType> hal_struct_types(const tt::tt_metal::Hal& hal);

// ============ Enums ============

struct Enumerator {
    std::string_view name;
    uint32_t value;
};

// An enum's name and every value it names.
struct EnumType {
    std::string_view name;
    std::vector<Enumerator> enumerators;
};

template <typename E>
EnumType enum_type() {
    static_assert(
        sizeof(E) <= sizeof(uint32_t) && std::is_unsigned_v<std::underlying_type_t<E>>,
        "an enumerator's value must fit a uint32_t");
    EnumType out{enchantum::type_name<E>, {}};
    for (const auto& [value, name] : enchantum::entries<E>) {
        out.enumerators.push_back({name, static_cast<uint32_t>(value)});
    }
    return out;
}

// enchantum cannot reflect EDMStatus: its values lie far outside the range it scans. So its enumerators are listed
// here, once. edm_status_listed's switch has no default, so an enumerator missing from the list fails the build.
#define FABRIC_MANIFEST_EDM_STATUSES(X) \
    X(STARTED)                          \
    X(REMOTE_HANDSHAKE_COMPLETE)        \
    X(LOCAL_HANDSHAKE_COMPLETE)         \
    X(READY_FOR_TRAFFIC)                \
    X(TERMINATED)                       \
    X(INITIALIZATION_STARTED)           \
    X(TXQ_INITIALIZED)                  \
    X(STREAM_REG_INITIALIZED)           \
    X(DOWNSTREAM_EDM_SETUP_STARTED)     \
    X(EDM_VCS_SETUP_COMPLETE)           \
    X(WORKER_INTERFACES_INITIALIZED)    \
    X(ETHERNET_HANDSHAKE_COMPLETE)      \
    X(VCS_OPENED)                       \
    X(ROUTING_TABLE_INITIALIZED)        \
    X(INITIALIZATION_COMPLETE)

constexpr bool edm_status_listed(EDMStatus status) {
    switch (status) {
#define FABRIC_MANIFEST_EDM_STATUS_CASE(name) case EDMStatus::name:
        FABRIC_MANIFEST_EDM_STATUSES(FABRIC_MANIFEST_EDM_STATUS_CASE)
#undef FABRIC_MANIFEST_EDM_STATUS_CASE
        return true;
    }
    return false;
}

template <>
inline EnumType enum_type<EDMStatus>() {
#define FABRIC_MANIFEST_EDM_STATUS_ENUMERATOR(name) Enumerator{#name, EDMStatus::name},
    return {enchantum::type_name<EDMStatus>, {FABRIC_MANIFEST_EDM_STATUSES(FABRIC_MANIFEST_EDM_STATUS_ENUMERATOR)}};
#undef FABRIC_MANIFEST_EDM_STATUS_ENUMERATOR
}

#undef FABRIC_MANIFEST_EDM_STATUSES

// Every enum a field or a described struct names. The manifest writes each one's values (enum_type).
using DescribedEnums = std::tuple<
    EDMStatus,
    TerminationSignal,
    CoordinatedEriscContextSwitchState,
    eth_chan_directions,
    manifest::NocCmdBuf,
    RouterState,
    RouterCommand,
    DynamicStatistics>;

}  // namespace tt::tt_fabric::layout
