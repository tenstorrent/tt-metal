// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Interpretation: one value, read by typed read into its registered type, turned into what a person reads. An
// interpreter sees one element of one read and the manifest facts in its context, never bytes, another region, another
// round or another router: anything that needs two values is analysis. It never throws, whatever the value holds.

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

#include <enchantum/enchantum.hpp>
#include <nlohmann/json.hpp>

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_enum_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_type_registry.hpp"

namespace tt::tt_fabric::decode {

using json = nlohmann::ordered_json;

// What interpreting one value can need beyond the value: manifest facts, never another captured value.
struct InterpretContext {
    // The architecture's facts: heartbeat_protocol, fabric_telemetry_version, the core registers' reset masks.
    const manifest::Arch& arch;
    // The router the value is from, for its fields, such as each ERISC's telemetry_enabled and telemetry_stats_mask.
    const manifest::Router& router;
    // The region's path below the router's node, such as "fields/erisc0/edm_status" or "l1_areas/fabric_telemetry".
    std::string_view path;
    // Which element of the region the value is: CodeProfilingTimerResult element i is timer bit i.
    uint32_t element_index = 0;
    // The ERISC whose fields name the region, such as 0 for "fields/erisc0/code_profiling" or
    // "channels/senders/vc0/ch1/fields/erisc0/free_slots". Null for a region no ERISC's fields name: a ring buffer, the
    // credit counters, leftover L1, an arch area.
    std::optional<uint32_t> erisc;
};

// The ERISC in a field region's path: N for ".../fields/erisc<N>/<key>", as for_each_router_region spells a field
// region's path. Null for any other path.
std::optional<uint32_t> find_path_erisc(std::string_view path);

// The context for the region at `path` of `router`, its ERISC from the path (find_path_erisc), at element 0.
InterpretContext make_interpret_context(
    const manifest::Arch& arch, const manifest::Router& router, std::string_view path);

// -- Named functions: meaning analysis needs as well, as typed results. The interpreters call them, and so does
// analysis, which never reads JSON. Each takes its meaning from where the kernel does.

// The enumerator `value` names, in lower case, or null when it names none. EDMStatus through find_edm_status_name;
// every other enum through enchantum, which must reach its enumerators.
template <typename E>
std::optional<std::string> find_enum_name(E value) {
    if constexpr (std::is_same_v<E, EDMStatus>) {
        return find_edm_status_name(value);
    } else {
        static_assert(
            enchantum::count<E> > 0,
            "Enchantum finds no enumerator of E: list its names in fabric_decode_enum_names.hpp, as EDMStatus's");
        const std::string_view enumerator = enchantum::to_string(value);
        if (enumerator.empty()) {
            return std::nullopt;
        }
        return manifest::lower_name(enumerator);
    }
}

// Whether the peer's handshake reached this core: its local_value is MAGIC_HANDSHAKE_VALUE, which the peer writes.
bool is_handshake_complete(const erisc::datamover::handshake::handshake_info_t& info);

// Whether ERISC `erisc` of `router` writes `statistic` in the router's FabricTelemetry.
bool writes_telemetry_statistic(const manifest::Router& router, uint32_t erisc, DynamicStatistics statistic);

// How a registered type is shown. Each registered type specializes it with one of:
//
//   // One element of a region.
//   static json make_json(const T& value, const InterpretContext& context);
//   // The whole region at once, for a type that means nothing one element at a time (std::byte).
//   static json make_region_json(std::span<const T> elements, const InterpretContext& context);
//
// Each struct's interpreter names the members it shows, so a member added to the struct is left out until it's added
// there, and one renamed or removed fails to build.
template <typename T>
struct Interpreter;

// The bytes of a region shown as a whole: how many there are, and how many aren't zero. std::byte regions and
// inactive regions are both shown this way.
json summarize_bytes(std::span<const std::byte> bytes);

template <>
struct Interpreter<uint32_t> {
    static json make_json(uint32_t value, const InterpretContext& context);
};

template <>
struct Interpreter<std::byte> {
    static json make_region_json(std::span<const std::byte> elements, const InterpretContext& context);
};

template <>
struct Interpreter<EDMStatus> {
    static json make_json(EDMStatus value, const InterpretContext& context);
};

template <>
struct Interpreter<TerminationSignal> {
    static json make_json(TerminationSignal value, const InterpretContext& context);
};

template <>
struct Interpreter<connection_interface::ConnectionValue> {
    static json make_json(connection_interface::ConnectionValue value, const InterpretContext& context);
};

template <>
struct Interpreter<CoordinatedEriscContextSwitchState> {
    static json make_json(CoordinatedEriscContextSwitchState value, const InterpretContext& context);
};

template <>
struct Interpreter<erisc::datamover::handshake::handshake_info_t> {
    static json make_json(const erisc::datamover::handshake::handshake_info_t& value, const InterpretContext& context);
};

template <>
struct Interpreter<SenderChannelProducerCursor> {
    static json make_json(const SenderChannelProducerCursor& value, const InterpretContext& context);
};

template <>
struct Interpreter<EDMChannelWorkerLocationInfo> {
    static json make_json(const EDMChannelWorkerLocationInfo& value, const InterpretContext& context);
};

template <>
struct Interpreter<ChannelTrimmingOverrides> {
    static json make_json(const ChannelTrimmingOverrides& value, const InterpretContext& context);
};

template <>
struct Interpreter<CodeProfilingTimerResult> {
    static json make_json(const CodeProfilingTimerResult& value, const InterpretContext& context);
};

template <>
struct Interpreter<FabricTelemetry> {
    static json make_json(const FabricTelemetry& value, const InterpretContext& context);
};

// A placeholder: what the table holds means something only through the routing table analysis. Its bytes are still
// at the region's .bin offset.
template <>
struct Interpreter<routing_l1_info_t> {
    static json make_json(const routing_l1_info_t& value, const InterpretContext& context);
};

namespace detail {

// A packet header's members every header type shares, from PacketHeaderBase. The command fields and the route come
// with the route work.
json make_packet_header_json(uint16_t payload_size_bytes, NocSendType noc_send_type, uint8_t src_ch_id);

// Whether every byte of `value` is zero: a ring buffer slot no packet was ever written to, since routers zero every
// header at startup.
template <typename T>
bool is_all_zero(const T& value) {
    const auto bytes = std::bit_cast<std::array<std::byte, sizeof(T)>>(value);
    return std::ranges::all_of(bytes, [](std::byte byte) { return byte == std::byte{0}; });
}

// One ring buffer slot's header: null when it was never written, else its shared members.
template <typename Header>
json make_slot_json(const Header& header) {
    if (is_all_zero(header)) {
        return json(nullptr);
    }
    return make_packet_header_json(header.payload_size_bytes, header.noc_send_type, header.src_ch_id);
}

}  // namespace detail

// Every 1D header type fabric can pick.
template <uint32_t ExtensionWords>
struct Interpreter<LowLatencyPacketHeaderT<ExtensionWords>> {
    static json make_json(const LowLatencyPacketHeaderT<ExtensionWords>& value, const InterpretContext& /*context*/) {
        return detail::make_slot_json(value);
    }
};

// Every 2D header type fabric can pick.
template <int RouteBufferSize>
struct Interpreter<HybridMeshPacketHeaderT<RouteBufferSize>> {
    static json make_json(const HybridMeshPacketHeaderT<RouteBufferSize>& value, const InterpretContext& /*context*/) {
        return detail::make_slot_json(value);
    }
};

// Whether T has an interpreter: a make_json for one element, or a make_region_json for the whole region.
template <typename T>
concept Interpretable = requires(const T& value, const InterpretContext& context) {
    { Interpreter<T>::make_json(value, context) } -> std::same_as<json>;
} || requires(std::span<const T> elements, const InterpretContext& context) {
    { Interpreter<T>::make_region_json(elements, context) } -> std::same_as<json>;
};

namespace detail {

template <typename... Ts>
consteval bool are_interpretable(manifest::TypeList<Ts...> /*types*/) {
    return (Interpretable<Ts> && ...);
}

}  // namespace detail

// Ensures that every registered type has an Interpreter.
static_assert(
    detail::are_interpretable(manifest::RegisteredTypes{}),
    "Every registered type needs an Interpreter (fabric_decode_interpret.hpp)");

}  // namespace tt::tt_fabric::decode
