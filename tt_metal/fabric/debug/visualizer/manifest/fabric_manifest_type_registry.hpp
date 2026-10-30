// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The types a fabric manifest may name, and the one name each goes by.
//
// A type is registered if decode reads it out of captured bytes, either a region of L1, or a stream register's word.
// The manifest names a type only through get_manifest_type_name<T>(), which refuses a type that isn't registered, so
// the writer can't name a type decode can't read. Decode runs on the same build, maps each name back to its type
// through this list, and rebuilds each value from its bytes.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>

#include <hostdevcommon/fabric_common.h>
#include <hostdevcommon/fabric_telemetry_msgs.h>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt_stl/type_name.hpp>

#include "tt_metal/fabric/channel_trimming_import.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_handshake_types.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/code_profiling_types.hpp"
#include "tt_metal/fabric/packet_header_types.hpp"

namespace tt::tt_fabric::manifest {

// Carries a list of types as template arguments. It holds no data.
template <typename... Ts>
struct TypeList {};

namespace detail {

// Generic template definition for concatenating TypeLists.
template <typename... Lists>
struct concat_type_lists;

// Base case for concatenating TypeLists, when there is only one list the concatenation result is the same list.
template <typename... Ts>
struct concat_type_lists<TypeList<Ts...>> {
    using type = TypeList<Ts...>;
};

// Recursive case for concatenating TypeLists. This merges the first two lists, and then recurses with the result and
// the remaining lists. Example: concat_type_lists<TypeList<int>, TypeList<float, char>, TypeList<bool>>::type
// would be TypeList<int, float, char, bool>.
template <typename... As, typename... Bs, typename... Rest>
struct concat_type_lists<TypeList<As...>, TypeList<Bs...>, Rest...>
    : concat_type_lists<TypeList<As..., Bs...>, Rest...> {};

// Generic packet header template definition.
template <typename Indices1D, typename Indices2D>
struct packet_header_type_list;

// Produces a TypeList of all the packet header types that fabric can pick for a run.
// For example: if PKT_HDR_1D_EXTENSION_WORDS is {1, 2} and PKT_HDR_2D_ROUTE_BUFFER_SIZES is {3},
// the type list would be
// TypeList<LowLatencyPacketHeaderT<1>, LowLatencyPacketHeaderT<2>, HybridMeshPacketHeaderT<3>>.
template <std::size_t... I, std::size_t... J>
struct packet_header_type_list<std::index_sequence<I...>, std::index_sequence<J...>> {
    using type = TypeList<
        LowLatencyPacketHeaderT<PKT_HDR_1D_EXTENSION_WORDS[I]>...,
        HybridMeshPacketHeaderT<PKT_HDR_2D_ROUTE_BUFFER_SIZES[J]>...>;
};

}  // namespace detail

// Every header type fabric can pick for a run. Ring buffer slots start with a packet header, and are followed by their
// payload (left as raw bytes).
using PacketHeaderTypes = typename detail::packet_header_type_list<
    std::make_index_sequence<PKT_HDR_1D_EXTENSION_WORDS.size()>,
    std::make_index_sequence<PKT_HDR_2D_ROUTE_BUFFER_SIZES.size()>>::type;

// Every type a manifest may name. Edit this list to add one.
using RegisteredTypes = typename detail::concat_type_lists<
    TypeList<
        // Words: single words, the credit counter arrays, and the counts in stream registers.
        uint32_t,
        // Raw bytes: what the manifest doesn't type, such as leftover L1.
        std::byte,
        // Enums held in L1 or in a stream register.
        EDMStatus,
        TerminationSignal,
        connection_interface::ConnectionValue,
        CoordinatedEriscContextSwitchState,
        // Structs held in L1.
        erisc::datamover::handshake::handshake_info_t,
        SenderChannelProducerCursor,
        EDMChannelWorkerLocationInfo,
        ChannelTrimmingOverrides,
        CodeProfilingTimerResult,
        FabricTelemetry,
        routing_l1_info_t>,
    PacketHeaderTypes>::type;

// Fallback case if List is not a TypeList.
template <typename T, typename List>
inline constexpr bool is_in_list_v = false;

// Check if T is in the TypeList TypeList<Ts...>.
template <typename T, typename... Ts>
inline constexpr bool is_in_list_v<T, TypeList<Ts...>> = (std::is_same_v<T, Ts> || ...);

// Check if T is in the RegisteredTypes TypeList.
template <typename T>
inline constexpr bool is_registered_v = is_in_list_v<T, RegisteredTypes>;

// The name the manifest and decode both know T by: its name without namespaces, as the compiler spells it.
template <typename T>
inline constexpr std::string_view manifest_type_name = ttsl::short_type_name<T>;

// The compiler leaves a template argument that equals its default out of a type's name (LowLatencyPacketHeaderT<0>
// comes out as LowLatencyPacketHeaderT<>), so the registered types whose argument is the default are named here.
template <>
inline constexpr std::string_view manifest_type_name<LowLatencyPacketHeaderT<0>> = "LowLatencyPacketHeaderT<0>";

template <>
inline constexpr std::string_view manifest_type_name<HybridMeshPacketHeaderT<36>> = "HybridMeshPacketHeaderT<36>";

template <typename T>
constexpr std::string_view get_manifest_type_name() {
    static_assert(
        is_registered_v<T>,
        "The manifest names only registered types: add T to RegisteredTypes (fabric_manifest_type_registry.hpp)");
    return manifest_type_name<T>;
}

namespace detail {

template <typename... Ts>
consteval bool are_names_unique(TypeList<Ts...> /*types*/) {
    const std::array<std::string_view, sizeof...(Ts)> names{manifest_type_name<Ts>...};
    for (std::size_t i = 0; i < names.size(); ++i) {
        for (std::size_t j = i + 1; j < names.size(); ++j) {
            if (names[i] == names[j]) {
                return false;
            }
        }
    }
    return true;
}

// A name holding "<>" lost a template argument that equals its default. Catches drift if the default changes.
template <typename... Ts>
consteval bool are_template_arguments_named(TypeList<Ts...> /*types*/) {
    return ((manifest_type_name<Ts>.find("<>") == std::string_view::npos) && ...);
}

template <typename... Ts>
consteval bool are_trivially_copyable(TypeList<Ts...> /*types*/) {
    return (std::is_trivially_copyable_v<Ts> && ...);
}

// PacketHeaderTypes repeats which template each routing mode uses, so check it against fabric's own list.
consteval bool registers_every_packet_header() {
    bool all_registered = true;
    for_each_packet_header_type([&](auto type, const PacketHeaderConfig& /*config*/) {
        all_registered = all_registered && is_registered_v<typename decltype(type)::type>;
    });
    return all_registered;
}

}  // namespace detail

static_assert(
    detail::are_names_unique(RegisteredTypes{}),
    "Two registered types have the same name without their namespaces, so decode couldn't tell them apart");
static_assert(
    detail::are_template_arguments_named(RegisteredTypes{}),
    "A registered type's name lost a template argument that equals its default: name it next to "
    "manifest_type_name's specializations");
static_assert(
    detail::are_trivially_copyable(RegisteredTypes{}),
    "Decode rebuilds a registered type from its bytes, so every registered type must be trivially copyable");
static_assert(
    detail::registers_every_packet_header(),
    "A packet header type fabric can pick isn't in PacketHeaderTypes, so ring buffer slots couldn't name it");

}  // namespace tt::tt_fabric::manifest
