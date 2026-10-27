// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The types the manifest describes, built from the struct layouts (fabric_struct_layouts.hpp) and the enums its schemas
// name. In the manifest,types.structs holds a StructType per described struct, and types.enums an EnumType per
// described enum.

#pragma once

#include <cstdint>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <hostdevcommon/fabric_common.h>
#include <hostdevcommon/fabric_telemetry_msgs.h>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_struct_layouts.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/struct_layout.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"
#include "tt_metal/hw/inc/hostdev/fabric_edm_packet_header.hpp"

namespace tt::tt_fabric::manifest {

// ============ Structs ============

// The StructType of a described struct, viewing its layout's members.
template <layout::Described T>
StructType struct_type() {
    return {
        layout::detail::get_struct_name<T>(),
        sizeof(T),
        std::span<const layout::Member>(layout::StructLayout<T>::members)};
}

// A StructType for each struct in layout::DescribedStructs, in its order.
inline std::vector<StructType> described_struct_types() {
    return []<typename... Ts>(std::type_identity<std::tuple<Ts...>>) {
        return std::vector<StructType>{struct_type<Ts>()...};
    }(std::type_identity<layout::DescribedStructs>{});
}

// ============ Enums ============

// The EnumType of E, with every value enchantum finds.
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

// enchantum works by scanning a range of integers for enumerator names, and then matching the values to the names.
// EDMStatus cannot be reflected by enchantum as its enumerator values lie far outside the range it scans. Thus, we
// list the enumerators here.
// To ensure that we don't drift from the actual EDMStatus type, the edm_status_listed() ensures that all
// enumerators are listed.
// The X-macro is used to list the enumerators once, so that they can be used for both edm_status_listed() and
// enum_type<EDMStatus>().
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

// Validate that all enumerators in EDMStatus are listed in the X-macro.
constexpr bool edm_status_listed(EDMStatus status) {
    switch (status) {
#define FABRIC_MANIFEST_EDM_STATUS_CASE(name) case EDMStatus::name:
        FABRIC_MANIFEST_EDM_STATUSES(FABRIC_MANIFEST_EDM_STATUS_CASE)
#undef FABRIC_MANIFEST_EDM_STATUS_CASE
        return true;
    }
    return false;
}

// Specialization for EDMStatus, because enchantum cannot reflect its values.
template <>
inline EnumType enum_type<EDMStatus>() {
#define FABRIC_MANIFEST_EDM_STATUS_ENUMERATOR(name) Enumerator{#name, EDMStatus::name},
    return {enchantum::type_name<EDMStatus>, {FABRIC_MANIFEST_EDM_STATUSES(FABRIC_MANIFEST_EDM_STATUS_ENUMERATOR)}};
#undef FABRIC_MANIFEST_EDM_STATUS_ENUMERATOR
}

#undef FABRIC_MANIFEST_EDM_STATUSES

// Every enum a field or a described struct names. The manifest writes each one's values.
using DescribedEnums = std::tuple<
    EDMStatus,
    TerminationSignal,
    connection_interface::ConnectionValue,
    CoordinatedEriscContextSwitchState,
    eth_chan_directions,
    NocCmdBuf,
    RouterState,
    RouterCommand,
    DynamicStatistics>;

// An EnumType for each enum in DescribedEnums, in its order.
inline std::vector<EnumType> described_enum_types() {
    return []<typename... Es>(std::type_identity<std::tuple<Es...>>) {
        return std::vector<EnumType>{enum_type<Es>()...};
    }(std::type_identity<DescribedEnums>{});
}

}  // namespace tt::tt_fabric::manifest
