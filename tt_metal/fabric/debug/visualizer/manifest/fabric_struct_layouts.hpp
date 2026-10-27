// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <type_traits>

#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "tt_metal/fabric/channel_trimming_import.hpp"
#include "tt_metal/fabric/packet_header_types.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake_types.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/code_profiling_types.hpp"
#include "tt_metal/hw/inc/hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/struct_layout.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/struct_layout_helpers.hpp"

namespace tt::tt_fabric::layout {

// ============ Enums ============

// EDMStatus's values lie far outside the range enchantum scans, so its enumerators are listed here. The X-macro lists
// them once, for both the EnumLayout and edm_status_listed(), whose switch fails to build (-Wswitch, as an error) when
// EDMStatus gains an enumerator the list doesn't have.
#define FABRIC_LAYOUT_EDM_STATUSES(X) \
    X(STARTED)                        \
    X(REMOTE_HANDSHAKE_COMPLETE)      \
    X(LOCAL_HANDSHAKE_COMPLETE)       \
    X(READY_FOR_TRAFFIC)              \
    X(TERMINATED)                     \
    X(INITIALIZATION_STARTED)         \
    X(TXQ_INITIALIZED)                \
    X(STREAM_REG_INITIALIZED)         \
    X(DOWNSTREAM_EDM_SETUP_STARTED)   \
    X(EDM_VCS_SETUP_COMPLETE)         \
    X(WORKER_INTERFACES_INITIALIZED)  \
    X(ETHERNET_HANDSHAKE_COMPLETE)    \
    X(VCS_OPENED)                     \
    X(ROUTING_TABLE_INITIALIZED)      \
    X(INITIALIZATION_COMPLETE)

constexpr bool edm_status_listed(EDMStatus status) {
    switch (status) {
#define FABRIC_LAYOUT_EDM_STATUS_CASE(name) case EDMStatus::name:
        FABRIC_LAYOUT_EDM_STATUSES(FABRIC_LAYOUT_EDM_STATUS_CASE)
#undef FABRIC_LAYOUT_EDM_STATUS_CASE
        return true;
    }
    return false;
}

template <>
struct EnumLayout<EDMStatus> {
#define FABRIC_LAYOUT_EDM_STATUS_ENUMERATOR(name) Enumerator{#name, EDMStatus::name},
    static constexpr std::array enumerators = {FABRIC_LAYOUT_EDM_STATUSES(FABRIC_LAYOUT_EDM_STATUS_ENUMERATOR)};
#undef FABRIC_LAYOUT_EDM_STATUS_ENUMERATOR
};

#undef FABRIC_LAYOUT_EDM_STATUSES

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

// The 2D routing table, which decode applies to routing_l1_info_t's union bytes in a 2D run. Each region's live part is
// sized by the mesh shape (routing_l1_info_t's mesh_y_size and mesh_x_size), so decode reads the regions by rule.
template <>
struct StructLayout<route_table_2d_t> {
    static constexpr std::array members = {
        LAYOUT_BYTES(route_table_2d_t, action_vectors),
        LAYOUT_BYTES(route_table_2d_t, mcast_trees),
    };
};
static_assert(check_struct_layout<route_table_2d_t>());

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

// ============ Diagnostics ============

template <>
struct StructLayout<CodeProfilingTimerResult> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(CodeProfilingTimerResult, total_cycles),
        LAYOUT_MEMBER(CodeProfilingTimerResult, num_instances),
    };
};
static_assert(check_struct_layout<CodeProfilingTimerResult>());

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
static_assert(check_struct_layout<ChannelTrimmingOverrides>());

// ============ Packet header commands ============

template <>
struct StructLayout<NocUnicastCommandHeader> {
    static constexpr std::array members = {
        LAYOUT_MEMBER(NocUnicastCommandHeader, noc_address),
    };
};
static_assert(check_struct_layout<NocUnicastCommandHeader>());

template <>
struct StructLayout<NocUnicastInlineWriteCommandHeader> {
    using T = NocUnicastInlineWriteCommandHeader;
    static constexpr std::array members = {
        LAYOUT_MEMBER(T, noc_address),
        LAYOUT_MEMBER(T, value),
        LAYOUT_TAIL_PAD(T, value),
    };
};
static_assert(check_struct_layout<NocUnicastInlineWriteCommandHeader>());

template <>
struct StructLayout<NocUnicastAtomicIncCommandHeader> {
    using T = NocUnicastAtomicIncCommandHeader;
    static constexpr std::array members = {
        LAYOUT_MEMBER(T, noc_address),
        LAYOUT_MEMBER(T, val),
        LAYOUT_MEMBER(T, flush),
        LAYOUT_TAIL_PAD(T, flush),
    };
};
static_assert(check_struct_layout<NocUnicastAtomicIncCommandHeader>());

template <>
struct StructLayout<NocUnicastAtomicIncFusedCommandHeader> {
    using T = NocUnicastAtomicIncFusedCommandHeader;
    static constexpr std::array members = {
        LAYOUT_MEMBER(T, noc_address),
        LAYOUT_MEMBER(T, semaphore_noc_address),
        LAYOUT_MEMBER(T, val),
        LAYOUT_MEMBER(T, flush),
        LAYOUT_TAIL_PAD(T, flush),
    };
};
static_assert(check_struct_layout<NocUnicastAtomicIncFusedCommandHeader>());

template <>
struct StructLayout<NocUnicastScatterCommandHeader> {
    using T = NocUnicastScatterCommandHeader;
    static constexpr std::array members = {
        LAYOUT_MEMBER(T, noc_address),
        LAYOUT_MEMBER(T, chunk_size),
        LAYOUT_MEMBER(T, chunk_count),
        // A 2-bit NocScatterWriteChunkEncoding per chunk, chunk 0 in the lowest bits (set_chunk_encoding). Entries from
        // chunk_count on are unused.
        LAYOUT_BYTES(T, chunk_encoding),
    };
};
static_assert(check_struct_layout<NocUnicastScatterCommandHeader>());

template <>
struct StructLayout<NocMulticastCommandHeader> {
    using T = NocMulticastCommandHeader;
    static constexpr std::array members = {
        LAYOUT_MEMBER(T, address),
        LAYOUT_MEMBER(T, noc_x_start),
        LAYOUT_MEMBER(T, noc_y_start),
        LAYOUT_MEMBER(T, mcast_rect_size_x),
        LAYOUT_MEMBER(T, mcast_rect_size_y),
    };
};
static_assert(check_struct_layout<NocMulticastCommandHeader>());

template <>
struct StructLayout<NocMulticastAtomicIncCommandHeader> {
    using T = NocMulticastAtomicIncCommandHeader;
    static constexpr std::array members = {
        LAYOUT_MEMBER(T, address),
        LAYOUT_MEMBER(T, val),
        LAYOUT_MEMBER(T, noc_x_start),
        LAYOUT_MEMBER(T, noc_y_start),
        LAYOUT_MEMBER(T, size_x),
        LAYOUT_MEMBER(T, size_y),
    };
};
static_assert(check_struct_layout<NocMulticastAtomicIncCommandHeader>());

template <>
struct StructLayout<NocSparseMulticastWriteCommandHeader> {
    using T = NocSparseMulticastWriteCommandHeader;
    static constexpr std::array members = {
        LAYOUT_MEMBER(T, noc_address),
        LAYOUT_MEMBER(T, counts),
        LAYOUT_MEMBER(T, num_dests),
        LAYOUT_MEMBER(T, num_chips),
        LAYOUT_MEMBER(T, write_idx),
        LAYOUT_MEMBER(T, chip_idx),
    };
};
static_assert(check_struct_layout<NocSparseMulticastWriteCommandHeader>());

// Each send type's member, as the header's to_noc_* setters write them.
template <>
struct TaggedUnionLayout<NocCommandFields> {
    using Tag = NocSendType;
    static constexpr std::array cases = {
        LAYOUT_TAGGED_UNION_CASE(NocCommandFields, unicast_write, NOC_UNICAST_WRITE),
        LAYOUT_TAGGED_UNION_CASE(NocCommandFields, unicast_read, NOC_UNICAST_READ),
        LAYOUT_TAGGED_UNION_CASE(NocCommandFields, unicast_inline_write, NOC_UNICAST_INLINE_WRITE),
        LAYOUT_TAGGED_UNION_CASE(NocCommandFields, mcast_write, NOC_MULTICAST_WRITE),
        LAYOUT_TAGGED_UNION_CASE(NocCommandFields, unicast_seminc, NOC_UNICAST_ATOMIC_INC),
        LAYOUT_TAGGED_UNION_CASE(NocCommandFields, unicast_seminc_fused, NOC_FUSED_UNICAST_ATOMIC_INC),
        LAYOUT_TAGGED_UNION_CASE(NocCommandFields, mcast_seminc, NOC_MULTICAST_ATOMIC_INC),
        LAYOUT_TAGGED_UNION_CASE(NocCommandFields, unicast_scatter_write, NOC_UNICAST_SCATTER_WRITE),
        LAYOUT_TAGGED_UNION_CASE(NocCommandFields, sparse_mcast_write, NOC_SPARSE_MCAST_WRITE),
    };
};
static_assert(validate_tagged_union_cases<NocCommandFields>(TaggedUnionLayout<NocCommandFields>::cases));

// ============ Packet headers ============

// The router reads value and shifts out one hop per chip. When value runs out, the router refills it from
// route_buffer[0] and shifts the other words down (update_packet_header_for_next_hop).
template <uint32_t N>
struct StructLayout<LowLatencyRoutingFieldsT<N>> {
    using T = LowLatencyRoutingFieldsT<N>;
    static constexpr std::string_view name = template_name<"LowLatencyRoutingFieldsT", N>;
    static constexpr std::array members = {
        LAYOUT_MEMBER(T, value),
        LAYOUT_MEMBER(T, route_buffer),
    };
};

// With no extension words there is no route_buffer.
template <>
struct StructLayout<LowLatencyRoutingFieldsT<0>> {
    using T = LowLatencyRoutingFieldsT<0>;
    static constexpr std::string_view name = template_name<"LowLatencyRoutingFieldsT", 0>;
    static constexpr std::array members = {
        LAYOUT_MEMBER(T, value),
    };
};

// The headers derive from PacketHeaderBase<Derived> and add members, so they are not standard layout, and offsetof
// on them raises -Winvalid-offsetof, an error in this build. GCC and Clang give the right offsets for the
// non-virtual single inheritance the headers use. The warning is off only for the layouts below.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"

namespace detail {

// The members every header gets from PacketHeaderBase. The base takes one initializer in the header's T{...}.
template <typename H>
constexpr std::array<Member, 4> packet_header_base_members() {
    return filled_by_one_initializer(std::array{
        LAYOUT_TAGGED_UNION(H, command_fields, noc_send_type),
        LAYOUT_MEMBER(H, payload_size_bytes),
        LAYOUT_MEMBER(H, noc_send_type),
        LAYOUT_MEMBER(H, src_ch_id),
    });
}

}  // namespace detail

template <uint32_t N>
struct StructLayout<LowLatencyPacketHeaderT<N>> {
    using T = LowLatencyPacketHeaderT<N>;
    static constexpr std::string_view name = template_name<"LowLatencyPacketHeaderT", N>;
    static constexpr auto members = [] {
        if constexpr (N == 0) {
            // padding0 is uint8_t[0] here, so it has no bytes to list.
            return concat_members(
                detail::packet_header_base_members<T>(), std::array{LAYOUT_MEMBER(T, routing_fields)});
        } else {
            return concat_members(
                detail::packet_header_base_members<T>(),
                std::array{LAYOUT_MEMBER(T, routing_fields), LAYOUT_MEMBER(T, padding0)});
        }
    }();
};

template <int N>
struct StructLayout<HybridMeshPacketHeaderT<N>> {
    using T = HybridMeshPacketHeaderT<N>;
    static constexpr std::string_view name = template_name<"HybridMeshPacketHeaderT", N>;
    static constexpr auto members = concat_members(
        detail::packet_header_base_members<T>(),
        std::array{
            // Reserved in 2D, where routing reads route_buffer instead.
            LAYOUT_BYTES(T, routing_fields),
            LAYOUT_MEMBER(T, route_buffer),
        },
        // Two anonymous unions, each listed by the members that span it and filled by one initializer: the chip and
        // mesh
        // ids rather than dst_start_node_id, and mcast_params rather than mcast_params_64.
        filled_by_one_initializer(std::array{LAYOUT_MEMBER(T, dst_start_chip_id), LAYOUT_MEMBER(T, dst_start_mesh_id)}),
        filled_by_one_initializer(std::array{LAYOUT_MEMBER(T, mcast_params)}));
};

// Every header fabric can choose (packet_header_types.hpp), and its routing fields where they have a layout (the 1D
// headers'). They aren't in DescribedStructs, since a run describes only the header it uses.
static_assert([] {
    for_each_packet_header_type([]<typename Header>(std::type_identity<Header>, const PacketHeaderConfig&) {
        static_assert(check_struct_layout<Header>());
        using RoutingFields = decltype(Header::routing_fields);
        if constexpr (Described<RoutingFields>) {
            static_assert(check_struct_layout<RoutingFields>());
        }
    });
    return true;
}());

#pragma GCC diagnostic pop

// ============ All described structs ============

// Every struct described at compile time. The manifest writes one type entry per element, which describes the
// struct's members.
using DescribedStructs = std::tuple<
    WorkerXY,
    EDMChannelWorkerLocationInfo,
    SenderChannelProducerCursor,
    erisc::datamover::handshake::handshake_info_t,
    RouterStateManager,
    route_table_2d_t,
    routing_l1_info_t,
    RiscTimestampV2,
    BandwidthTelemetry,
    EriscDynamicEntry,
    DynamicInfo,
    StaticInfo,
    FabricTelemetry,
    CodeProfilingTimerResult,
    ChannelTrimmingOverrides,
    NocUnicastCommandHeader,
    NocUnicastInlineWriteCommandHeader,
    NocUnicastAtomicIncCommandHeader,
    NocUnicastAtomicIncFusedCommandHeader,
    NocUnicastScatterCommandHeader,
    NocMulticastCommandHeader,
    NocMulticastAtomicIncCommandHeader,
    NocSparseMulticastWriteCommandHeader>;

// ============ All described tagged unions ============

// Every tagged union described at compile time. The manifest writes one type entry per element, beside the structs,
// which describes the union's tag enum and cases.
using DescribedTaggedUnions = std::tuple<NocCommandFields>;

}  // namespace tt::tt_fabric::layout
