// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_interpret.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstddef>
#include <iterator>
#include <system_error>
#include <type_traits>
#include <variant>
#include <vector>

#include <hostdevcommon/fabric_common.h>
#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_fields.hpp"

namespace tt::tt_fabric::decode {

namespace {

// A bitmask of a flag enum's enumerators: its number, the names of the one-bit enumerators set in it, and, only when
// some are set, the bits no enumerator names.
template <typename E>
json make_flags_json(E value) {
    using Underlying = std::underlying_type_t<E>;
    const auto bits = static_cast<Underlying>(value);
    Underlying named_bits = 0;
    json names = json::array();
    for (const E flag : enchantum::values<E>) {
        const auto flag_bits = static_cast<Underlying>(flag);
        if (std::has_single_bit(flag_bits) && (bits & flag_bits) != 0) {
            names.push_back(*find_enum_name(flag));
            named_bits |= flag_bits;
        }
    }
    json out;
    out["value"] = bits;
    out["names"] = std::move(names);
    if (const auto unnamed_bits = static_cast<Underlying>(bits & ~named_bits); unnamed_bits != 0) {
        out["unnamed_bits"] = unnamed_bits;
    }
    return out;
}

// A C array member, as a JSON array.
template <typename T, size_t N>
json make_array_json(const T (&values)[N]) {
    return json(std::vector<T>(std::begin(values), std::end(values)));
}

// A number or flag field's value: a flag is 0 or 1. Null for a field of another kind.
std::optional<uint32_t> read_field_value(const manifest::Field& field) {
    if (!field.content.has_value()) {
        return std::nullopt;
    }
    if (const auto* number = std::get_if<manifest::content::Number>(&*field.content)) {
        return number->value;
    }
    if (const auto* flag = std::get_if<manifest::content::Flag>(&*field.content)) {
        return flag->value ? 1 : 0;
    }
    return std::nullopt;
}

// The value of the number or flag field `key` on ERISC `erisc` of `router`. Null when it isn't there.
std::optional<uint32_t> find_erisc_field_value(
    const manifest::Router& router, uint32_t erisc, manifest::RouterFieldKey key) {
    const manifest::Field* field = manifest::find_erisc_field(router.fields_by_erisc, erisc, key);
    return field == nullptr ? std::nullopt : read_field_value(*field);
}

// Whether any of the router's ERISCs writes `statistic`.
bool any_erisc_writes_telemetry_statistic(const manifest::Router& router, DynamicStatistics statistic) {
    for (uint32_t erisc = 0; erisc < router.eriscs.size(); ++erisc) {
        if (writes_telemetry_statistic(router, erisc, statistic)) {
            return true;
        }
    }
    return false;
}

// A value the kernel writes only when `written`: the value's JSON, or null.
json make_if_written_json(bool written, json value) { return written ? std::move(value) : json(nullptr); }

// A 64-bit timestamp or counter the kernel keeps as two words.
uint64_t read_timestamp(const RiscTimestampV2& timestamp) { return timestamp.full; }

json make_bandwidth_json(const BandwidthTelemetry& bandwidth) {
    json out;
    out["elapsed_active_cycles"] = read_timestamp(bandwidth.elapsed_active_cycles);
    out["elapsed_cycles"] = read_timestamp(bandwidth.elapsed_cycles);
    out["num_words_sent"] = bandwidth.num_words_sent;
    out["num_packets_sent"] = bandwidth.num_packets_sent;
    return out;
}

// ERISC `erisc`'s entry: each statistic it writes, and null for each it doesn't.
json make_erisc_dynamic_entry_json(const EriscDynamicEntry& entry, const manifest::Router& router, uint32_t erisc) {
    const auto if_written = [&](DynamicStatistics statistic, json value) {
        return make_if_written_json(writes_telemetry_statistic(router, erisc, statistic), std::move(value));
    };
    json out;
    out["router_state"] = if_written(DynamicStatistics::ROUTER_STATE, detail::make_enum_json(entry.router_state));
    out["tx_heartbeat"] = if_written(DynamicStatistics::HEARTBEAT_TX, read_timestamp(entry.tx_heartbeat));
    out["rx_heartbeat"] = if_written(DynamicStatistics::HEARTBEAT_RX, read_timestamp(entry.rx_heartbeat));
    return out;
}

json make_static_info_json(const StaticInfo& info) {
    json out;
    out["version"] = info.version;
    out["mesh_id"] = info.mesh_id;
    out["neighbor_mesh_id"] = info.neighbor_mesh_id;
    out["device_id"] = info.device_id;
    out["neighbor_device_id"] = info.neighbor_device_id;
    out["direction"] = info.direction;
    out["supported_stats"] = make_flags_json(info.supported_stats);
    out["fabric_config"] = info.fabric_config;
    return out;
}

// The dynamic statistics, with null for each one the router's ERISCs don't write. The bandwidth counts are the
// router's, not one ERISC's, so they're written when any ERISC writes them.
json make_dynamic_info_json(const DynamicInfo& info, const manifest::Router& router) {
    const bool bandwidth_written = any_erisc_writes_telemetry_statistic(router, DynamicStatistics::BANDWIDTH);
    json out;
    out["tx_bandwidth"] = make_if_written_json(bandwidth_written, make_bandwidth_json(info.tx_bandwidth));
    out["rx_bandwidth"] = make_if_written_json(bandwidth_written, make_bandwidth_json(info.rx_bandwidth));
    json eriscs = json::array();
    for (uint32_t erisc = 0; erisc < std::size(info.erisc); ++erisc) {
        eriscs.push_back(make_erisc_dynamic_entry_json(info.erisc[erisc], router, erisc));
    }
    out["erisc"] = std::move(eriscs);
    return out;
}

static_assert(sizeof(FabricTelemetry) % sizeof(uint32_t) == 0, "FabricTelemetry is shown raw as whole words");

// A FabricTelemetry of another layout than this build's: its version, and its words as they are.
json make_raw_telemetry_json(const FabricTelemetry& value) {
    const auto words = std::bit_cast<std::array<uint32_t, sizeof(FabricTelemetry) / sizeof(uint32_t)>>(value);
    json out;
    out["version"] = value.static_info.version;
    out["raw"] = words;
    return out;
}

// The code profiling timer element `index` holds: the CodeProfilingTimerType whose bit is `index`.
json make_code_profiling_timer_json(uint32_t index) {
    if (index >= sizeof(uint32_t) * 8) {
        return json(nullptr);
    }
    return detail::make_enum_json(static_cast<CodeProfilingTimerType>(uint32_t{1} << index));
}

}  // namespace

std::optional<uint32_t> find_path_erisc(std::string_view path) {
    // for_each_router_region spells a field region's path "<fields path>/erisc<N>/<key>", where the fields path is
    // "fields" or ends in "/fields".
    constexpr std::string_view k_erisc_marker = "fields/erisc";
    const size_t marker = path.find(k_erisc_marker);
    if (marker == std::string_view::npos || (marker != 0 && path[marker - 1] != '/')) {
        return std::nullopt;
    }
    const std::string_view rest = path.substr(marker + k_erisc_marker.size());
    uint32_t erisc = 0;
    const auto [end, error] = std::from_chars(rest.data(), rest.data() + rest.size(), erisc);
    if (error != std::errc{} || end == rest.data() || end == rest.data() + rest.size() || *end != '/') {
        return std::nullopt;
    }
    return erisc;
}

InterpretContext make_interpret_context(
    const manifest::ManifestFacts& facts,
    const RouteView& routes,
    const manifest::RouterRef& ref,
    const manifest::Router& router,
    std::string_view path) {
    const manifest::Mesh* mesh = manifest::find_mesh(facts, ref.node.mesh_id);
    TT_FATAL(mesh != nullptr, "Fabric decode: the manifest has no mesh {} for router {}", *ref.node.mesh_id, path);
    return InterpretContext{
        .arch = facts.arch,
        .fabric = facts.fabric,
        .mesh = *mesh,
        .router = router,
        .router_ref = ref,
        .routes = routes,
        .path = path,
        .erisc = find_path_erisc(path)};
}

bool is_handshake_complete(const erisc::datamover::handshake::handshake_info_t& info) {
    return info.local_value == erisc::datamover::handshake::MAGIC_HANDSHAKE_VALUE;
}

bool writes_telemetry_statistic(const manifest::Router& router, uint32_t erisc, DynamicStatistics statistic) {
    const std::optional<uint32_t> enabled = find_erisc_field_value(router, erisc, "telemetry_enabled");
    const std::optional<uint32_t> stats_mask = find_erisc_field_value(router, erisc, "telemetry_stats_mask");
    if (!enabled.has_value() || !stats_mask.has_value()) {
        return false;
    }
    return is_telemetry_statistic_written(*enabled != 0, static_cast<uint8_t>(*stats_mask), statistic);
}

json summarize_bytes(std::span<const std::byte> bytes) {
    json out;
    out["size_bytes"] = bytes.size();
    out["nonzero_bytes"] = std::ranges::count_if(bytes, [](std::byte byte) { return byte != std::byte{0}; });
    return out;
}

json Interpreter<uint32_t>::make_json(uint32_t value, const InterpretContext& /*context*/) { return json(value); }

json Interpreter<std::byte>::make_region_json(
    std::span<const std::byte> elements, const InterpretContext& /*context*/) {
    return summarize_bytes(elements);
}

json Interpreter<EDMStatus>::make_json(EDMStatus value, const InterpretContext& /*context*/) {
    return detail::make_enum_json(value);
}

json Interpreter<TerminationSignal>::make_json(TerminationSignal value, const InterpretContext& /*context*/) {
    return detail::make_enum_json(value);
}

json Interpreter<connection_interface::ConnectionValue>::make_json(
    connection_interface::ConnectionValue value, const InterpretContext& /*context*/) {
    return detail::make_enum_json(value);
}

json Interpreter<CoordinatedEriscContextSwitchState>::make_json(
    CoordinatedEriscContextSwitchState value, const InterpretContext& /*context*/) {
    return detail::make_enum_json(value);
}

json Interpreter<erisc::datamover::handshake::handshake_info_t>::make_json(
    const erisc::datamover::handshake::handshake_info_t& value, const InterpretContext& /*context*/) {
    json out;
    out["local_value"] = value.local_value;
    out["complete"] = is_handshake_complete(value);
    out["neighbor_mesh_id"] = value.neighbor_mesh_id;
    out["neighbor_device_id"] = value.neighbor_device_id;
    out["scratch"] = make_array_json(value.scratch);
    return out;
}

json Interpreter<SenderChannelProducerCursor>::make_json(
    const SenderChannelProducerCursor& value, const InterpretContext& /*context*/) {
    json out;
    out["write_counter"] = value.write_counter;
    out["write_index"] = value.write_index;
    return out;
}

json Interpreter<EDMChannelWorkerLocationInfo>::make_json(
    const EDMChannelWorkerLocationInfo& value, const InterpretContext& /*context*/) {
    json out;
    out["worker_semaphore_address"] = value.worker_semaphore_address;
    out["worker_teardown_semaphore_address"] = value.worker_teardown_semaphore_address;
    out["worker_xy"] = {{"x", value.worker_xy.x}, {"y", value.worker_xy.y}};
    out["edm_read_counter"] = value.edm_read_counter;
    return out;
}

json Interpreter<ChannelTrimmingOverrides>::make_json(
    const ChannelTrimmingOverrides& value, const InterpretContext& /*context*/) {
    json out;
    out["sender_channel_min_packet_size_seen_bytes_by_vc"] = value.sender_channel_min_packet_size_seen_bytes_by_vc;
    out["sender_channel_max_packet_size_seen_bytes_by_vc"] = value.sender_channel_max_packet_size_seen_bytes_by_vc;
    out["sender_channel_used_bitfield_by_vc"] = value.sender_channel_used_bitfield_by_vc;
    out["sender_channel_forwarded_to_bitfield_by_vc"] = value.sender_channel_forwarded_to_bitfield_by_vc;
    out["receiver_channel_data_forwarded_bitfield_by_vc"] = value.receiver_channel_data_forwarded_bitfield_by_vc;
    out["used_noc_send_type_by_vc_bitfield"] = value.used_noc_send_type_by_vc_bitfield;
    return out;
}

json Interpreter<RouterChannelProgressLog>::make_json(
    const RouterChannelProgressLog& value, const InterpretContext& /*context*/) {
    json out;
    json& senders = out["sender_channels"] = json::array();
    for (const SenderChannelProgressRecord& sender : value.sender_channels) {
        senders.push_back({
            {"next_slot_addr", sender.next_slot_addr},
            {"sent", sender.sent},
            {"completed", sender.completed},
            {"first_level_acked", sender.first_level_acked},
        });
    }
    json& receivers = out["receiver_channels"] = json::array();
    for (const ReceiverChannelProgressRecord& receiver : value.receiver_channels) {
        receivers.push_back({
            {"next_index", receiver.next_index},
            {"uncredited", receiver.uncredited},
            {"packets_ready", receiver.packets_ready},
        });
    }
    out["sender_pov_receiver_free_slots_per_vc"] = value.sender_pov_receiver_free_slots_per_vc;
    out["receiver_pov_sender_next_slot_addr_by_vc"] = value.receiver_pov_sender_next_slot_addr_by_vc;
    out["relay_next_slot_addr"] = value.relay_next_slot_addr;
    return out;
}

// Element i of the region is timer bit i. Whether that timer is enabled is its bit in the ERISC's
// code_profiling_timers field, written only when the context names the ERISC.
json Interpreter<CodeProfilingTimerResult>::make_json(
    const CodeProfilingTimerResult& value, const InterpretContext& context) {
    json out;
    out["timer"] = make_code_profiling_timer_json(context.element_index);
    if (context.erisc.has_value() && context.element_index < sizeof(uint32_t) * 8) {
        const std::optional<uint32_t> timers =
            find_erisc_field_value(context.router, *context.erisc, "code_profiling_timers");
        if (timers.has_value()) {
            out["enabled"] = (*timers & (uint32_t{1} << context.element_index)) != 0;
        }
    }
    out["total_cycles"] = value.total_cycles;
    out["num_instances"] = value.num_instances;
    return out;
}

json Interpreter<FabricTelemetry>::make_json(const FabricTelemetry& value, const InterpretContext& context) {
    // Another version is another layout, so none of its members can be read by this build's names.
    if (value.static_info.version != context.arch.fabric_telemetry_version) {
        return make_raw_telemetry_json(value);
    }
    json out;
    out["static_info"] = make_static_info_json(value.static_info);
    out["dynamic_info"] = make_dynamic_info_json(value.dynamic_info, context.router);
    out["postcode"] = detail::make_enum_json(static_cast<EDMStatus>(value.postcode));
    return out;
}

// Only what is the router core's own: the rest of the table is the chip's, written by make_chip_routing_table_json.
json Interpreter<routing_l1_info_t>::make_json(const routing_l1_info_t& value, const InterpretContext& /*context*/) {
    json out;
    out["state"] = detail::make_enum_json(value.state_manager.state);
    out["command"] = detail::make_enum_json(value.state_manager.command);
    return out;
}

}  // namespace tt::tt_fabric::decode
