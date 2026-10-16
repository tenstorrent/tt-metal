// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The layouts of the packet headers FabricContext can choose.

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_struct_layouts.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/manifest/struct_layout.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/hw/inc/hostdev/fabric_edm_packet_header.hpp"

namespace tt::tt_fabric::layout {

// ============ 1D routing fields ============

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

// ============ Packet headers ============

// The headers derive from PacketHeaderBase<Derived> and add members, so they are not standard layout, and offsetof
// on them raises -Winvalid-offsetof, an error in this build. GCC and Clang give the right offsets for the
// non-virtual single inheritance the headers use. The warning is off only for the layouts below.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"

namespace {

// The members every header has from PacketHeaderBase.
template <typename H>
constexpr std::array<Member, 4> packet_header_base_members() {
    return {
        LAYOUT_TAGGED_UNION(H, command_fields, noc_send_type),
        LAYOUT_MEMBER(H, payload_size_bytes),
        LAYOUT_MEMBER(H, noc_send_type),
        LAYOUT_MEMBER(H, src_ch_id),
    };
}

// The members of HybridMeshPacketHeaderT, for it and for the UDM header derived from it.
template <typename H>
constexpr auto hybrid_mesh_members() {
    return concat_members(
        packet_header_base_members<H>(),
        std::array{
            // Reserved in 2D, where routing reads route_buffer instead.
            LAYOUT_PAD(H, routing_fields),
            LAYOUT_MEMBER(H, route_buffer),
            // Two anonymous unions, each listed by the members that span it: the chip and mesh ids rather than
            // dst_start_node_id, and mcast_params rather than mcast_params_64.
            LAYOUT_MEMBER(H, dst_start_chip_id),
            LAYOUT_MEMBER(H, dst_start_mesh_id),
            LAYOUT_MEMBER(H, mcast_params),
        });
}

}  // namespace

template <uint32_t N>
struct StructLayout<LowLatencyPacketHeaderT<N>> {
    using T = LowLatencyPacketHeaderT<N>;
    static constexpr std::string_view name = template_name<"LowLatencyPacketHeaderT", N>;
    static constexpr auto members = [] {
        if constexpr (N == 0) {
            // padding0 is uint8_t[0] here, so it has no bytes to list.
            return concat_members(packet_header_base_members<T>(), std::array{LAYOUT_MEMBER(T, routing_fields)});
        } else {
            return concat_members(
                packet_header_base_members<T>(), std::array{LAYOUT_MEMBER(T, routing_fields), LAYOUT_PAD(T, padding0)});
        }
    }();
};

template <int N>
struct StructLayout<HybridMeshPacketHeaderT<N>> {
    static constexpr std::string_view name = template_name<"HybridMeshPacketHeaderT", N>;
    static constexpr auto members = hybrid_mesh_members<HybridMeshPacketHeaderT<N>>();
};

template <int N>
struct StructLayout<UDMHybridMeshPacketHeaderT<N>> {
    using T = UDMHybridMeshPacketHeaderT<N>;
    static constexpr std::string_view name = template_name<"UDMHybridMeshPacketHeaderT", N>;
    static constexpr auto members =
        concat_members(hybrid_mesh_members<T>(), std::array{LAYOUT_TAGGED_UNION(T, udm_control, noc_send_type)});
};

#pragma GCC diagnostic pop

// ============ Every header FabricContext can choose ============

namespace {

// A header and the structs only it holds, which DescribedStructs leaves out: a 1D header's routing fields.
template <typename H>
struct PacketHeaderStructs {
    using type = std::tuple<H>;
};
template <uint32_t N>
struct PacketHeaderStructs<LowLatencyPacketHeaderT<N>> {
    using type = std::tuple<LowLatencyPacketHeaderT<N>, LowLatencyRoutingFieldsT<N>>;
};

// Fails the build, naming T, unless T's members cover every byte of it once, in order.
template <typename T>
constexpr bool layout_covers_struct() {
    static_assert(validate_struct_members<T>(StructLayout<T>::members), "a layout must list every byte once, in order");
    return true;
}

// Fails the build, naming H, unless H and the structs only it holds have valid layouts. A vtable pointer would shift
// every member, by a different amount on the host than on the device, so a header must not be polymorphic.
template <typename H>
constexpr bool packet_header_layouts_are_valid() {
    static_assert(!std::is_polymorphic_v<H>, "a packet header must not be polymorphic");
    return []<typename... Ts>(std::type_identity<std::tuple<Ts...>>) {
        return (layout_covers_struct<Ts>() && ...);
    }(std::type_identity<typename PacketHeaderStructs<H>::type>{});
}

}  // namespace

// Every header FabricContext can choose, so a size added to its lists gets a checked layout with no other edit.
static_assert([]<std::size_t... I>(std::index_sequence<I...>) {
    return (
        packet_header_layouts_are_valid<typename decltype(packet_header_detail::packet_header_type_at<I>())::type>() &&
        ...);
}(std::make_index_sequence<PACKET_HEADER_CONFIGS.size()>{}));

Type packet_header_type(const PacketHeaderConfig& config) {
    return visit_packet_header_type(config, []<typename H>(std::type_identity<H>) { return type_of<H>(); });
}

std::vector<StructType> packet_header_struct_types(const PacketHeaderConfig& config) {
    return visit_packet_header_type(config, []<typename H>(std::type_identity<H>) {
        return []<typename... Ts>(std::type_identity<std::tuple<Ts...>>) {
            return std::vector<StructType>{struct_type<Ts>()...};
        }(std::type_identity<typename PacketHeaderStructs<H>::type>{});
    });
}

}  // namespace tt::tt_fabric::layout
