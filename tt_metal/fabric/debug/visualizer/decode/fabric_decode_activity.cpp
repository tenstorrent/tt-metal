// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_activity.hpp"

#include <algorithm>
#include <array>
#include <string_view>
#include <type_traits>

#include <enchantum/enchantum.hpp>
#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_regions.hpp"

namespace tt::tt_fabric::decode {

namespace {

// The paths of the regions the manifest marks with each role, arch areas first.
struct MultiSampledPaths {
    std::vector<std::string> loop_heartbeat;
    std::vector<std::string> fabric_telemetry;
};

MultiSampledPaths find_multi_sampled_paths(const manifest::Router& router, const manifest::Arch& arch) {
    MultiSampledPaths paths;
    const auto add = [&](std::string path, const std::optional<manifest::MultiSampleRole>& role) {
        if (!role.has_value()) {
            return;
        }
        switch (*role) {
            case manifest::MultiSampleRole::LOOP_HEARTBEAT: paths.loop_heartbeat.push_back(std::move(path)); return;
            case manifest::MultiSampleRole::FABRIC_TELEMETRY: paths.fabric_telemetry.push_back(std::move(path)); return;
        }
        TT_THROW("Fabric decode: unknown multi-sample role {}", static_cast<int>(*role));
    };
    manifest::for_each_arch_area(arch, [&](std::string_view name, const manifest::content::L1& area) {
        add(make_arch_area_path(name), area.multi_sampled);
    });
    manifest::for_each_router_region(
        router, [&](std::string_view path, const auto& region) { add(std::string(path), region.multi_sampled); });
    return paths;
}

// The one path with `role`, or null when the manifest marks none.
std::optional<std::string> find_single_path(const std::vector<std::string>& paths, manifest::MultiSampleRole role) {
    TT_FATAL(
        paths.size() <= 1,
        "Fabric decode: the manifest marks {} regions {}, but activity reads only one",
        paths.size(),
        manifest::lower_enum_name(role));
    return paths.empty() ? std::nullopt : std::optional(paths.front());
}

// Whether any two values differ. Null with fewer than two.
std::optional<bool> find_whether_moved(const std::vector<uint64_t>& values) {
    if (values.size() < 2) {
        return std::nullopt;
    }
    return std::ranges::any_of(values, [&](uint64_t value) { return value != values.front(); });
}

LoopHeartbeat read_loop_heartbeat(
    const RouterValues& values, const std::optional<std::string>& path, const manifest::HeartbeatProtocol& protocol) {
    LoopHeartbeat heartbeat;
    if (!path.has_value()) {
        return heartbeat;
    }
    heartbeat.path = *path;
    const RegionReads<uint32_t>* reads = values.find_reads<uint32_t>(*path);
    if (reads == nullptr) {
        return heartbeat;
    }
    std::vector<uint64_t> counts;
    for (const RegionRead<uint32_t>& read : *reads) {
        const uint32_t word = read.elements.front();
        heartbeat.reads.push_back(HeartbeatRead{.round = read.round, .t_ns = read.t_ns, .value = word});
        if (is_router_heartbeat(word, protocol)) {
            counts.push_back(get_heartbeat_count(word, protocol));
        }
    }
    heartbeat.moved = find_whether_moved(counts);
    return heartbeat;
}

// The heartbeat `side` of entry `erisc` in a FabricTelemetry.
uint64_t read_telemetry_heartbeat(const FabricTelemetry& telemetry, uint32_t erisc, HeartbeatSide side) {
    const EriscDynamicEntry& entry = telemetry.dynamic_info.erisc[erisc];
    return side == HeartbeatSide::TX ? entry.tx_heartbeat.full : entry.rx_heartbeat.full;
}

// The statistic `side`'s heartbeat is.
DynamicStatistics get_heartbeat_statistic(HeartbeatSide side) {
    return side == HeartbeatSide::TX ? DynamicStatistics::HEARTBEAT_TX : DynamicStatistics::HEARTBEAT_RX;
}

// Each side of each of the router's ERISCs, from the telemetry area's reads.
std::vector<TelemetryHeartbeat> read_telemetry_heartbeats(
    const RouterValues& values,
    const std::optional<std::string>& path,
    const manifest::Router& router,
    const manifest::Arch& arch) {
    const RegionReads<FabricTelemetry>* reads =
        path.has_value() ? values.find_reads<FabricTelemetry>(*path) : nullptr;
    // A read of another layout can't be read by this build's members.
    const auto is_this_layout = [&](const RegionRead<FabricTelemetry>& read) {
        return read.elements.front().static_info.version == arch.fabric_telemetry_version;
    };
    const bool layout_known = reads != nullptr && std::ranges::all_of(*reads, is_this_layout);
    constexpr uint32_t k_num_telemetry_eriscs = std::extent_v<decltype(DynamicInfo::erisc)>;
    const uint32_t num_eriscs = std::min<uint32_t>(router.eriscs.size(), k_num_telemetry_eriscs);

    std::vector<TelemetryHeartbeat> heartbeats;
    for (uint32_t erisc = 0; erisc < num_eriscs; ++erisc) {
        for (const HeartbeatSide side : enchantum::values<HeartbeatSide>) {
            TelemetryHeartbeat heartbeat{.erisc = erisc, .side = side};
            heartbeat.written =
                layout_known && writes_telemetry_statistic(router, erisc, get_heartbeat_statistic(side));
            if (heartbeat.written) {
                std::vector<uint64_t> heartbeat_values;
                for (const RegionRead<FabricTelemetry>& read : *reads) {
                    const uint64_t value = read_telemetry_heartbeat(read.elements.front(), erisc, side);
                    heartbeat.reads.push_back(HeartbeatRead{.round = read.round, .t_ns = read.t_ns, .value = value});
                    heartbeat_values.push_back(value);
                }
                heartbeat.moved = find_whether_moved(heartbeat_values);
            }
            heartbeats.push_back(std::move(heartbeat));
        }
    }
    return heartbeats;
}

// The verdict, from the loop heartbeat and the telemetry heartbeats the router writes.
void decide_verdict(RouterActivityVerdict& activity, const manifest::HeartbeatProtocol& protocol) {
    const LoopHeartbeat& loop = activity.loop_heartbeat;
    const auto num_router_reads = std::ranges::count_if(
        loop.reads, [&](const HeartbeatRead& read) { return is_router_heartbeat(read.value, protocol); });
    if (num_router_reads == 0 && loop.reads.size() >= 2) {
        // Read often enough, but never the router's: base firmware's, or nothing's.
        activity.verdict = RouterActivity::NOT_LOOPING;
        return;
    }
    if (!loop.moved.has_value()) {
        activity.verdict = RouterActivity::UNKNOWN;
        return;
    }
    if (!*loop.moved) {
        activity.verdict = RouterActivity::NOT_LOOPING;
        return;
    }
    // The loop runs: the telemetry heartbeats the router writes say whether it makes progress.
    bool any_written = false;
    bool all_written_moved = true;
    for (const TelemetryHeartbeat& heartbeat : activity.telemetry_heartbeats) {
        if (!heartbeat.written) {
            continue;
        }
        any_written = true;
        if (!heartbeat.moved.has_value()) {
            all_written_moved = false;
        } else if (!*heartbeat.moved) {
            activity.stopped.emplace_back(heartbeat.erisc, heartbeat.side);
        }
    }
    if (!activity.stopped.empty()) {
        activity.verdict = RouterActivity::STUCK;
    } else if (any_written && all_written_moved) {
        activity.verdict = RouterActivity::RUNNING;
    } else {
        activity.verdict = RouterActivity::LOOPING;
    }
}

json make_round_json(const std::optional<uint32_t>& round) { return round.has_value() ? json(*round) : json(nullptr); }

json make_loop_heartbeat_json(const LoopHeartbeat& heartbeat, const manifest::HeartbeatProtocol& protocol) {
    json out;
    out["path"] = heartbeat.path.empty() ? json(nullptr) : json(heartbeat.path);
    json reads = json::array();
    for (const HeartbeatRead& read : heartbeat.reads) {
        const auto word = static_cast<uint32_t>(read.value);
        json read_json;
        read_json["round"] = make_round_json(read.round);
        read_json["t_ns"] = read.t_ns;
        read_json["word"] = word;
        read_json["routers"] = is_router_heartbeat(word, protocol);
        if (is_router_heartbeat(word, protocol)) {
            read_json["count"] = get_heartbeat_count(word, protocol);
        }
        reads.push_back(std::move(read_json));
    }
    out["reads"] = std::move(reads);
    out["moved"] = heartbeat.moved.has_value() ? json(*heartbeat.moved) : json(nullptr);
    return out;
}

json make_telemetry_heartbeat_json(const TelemetryHeartbeat& heartbeat) {
    json out;
    out["erisc"] = heartbeat.erisc;
    out["side"] = manifest::lower_enum_name(heartbeat.side);
    out["written"] = heartbeat.written;
    if (!heartbeat.written) {
        return out;
    }
    json reads = json::array();
    for (const HeartbeatRead& read : heartbeat.reads) {
        json read_json;
        read_json["round"] = make_round_json(read.round);
        read_json["t_ns"] = read.t_ns;
        read_json["value"] = read.value;
        reads.push_back(std::move(read_json));
    }
    out["reads"] = std::move(reads);
    out["moved"] = heartbeat.moved.has_value() ? json(*heartbeat.moved) : json(nullptr);
    return out;
}

}  // namespace

bool is_router_heartbeat(uint32_t word, const manifest::HeartbeatProtocol& protocol) {
    return (word & protocol.magic_mask) == protocol.magic;
}

uint32_t get_heartbeat_count(uint32_t word, const manifest::HeartbeatProtocol& protocol) {
    return word & ~protocol.magic_mask;
}

RouterActivityVerdict judge_router_activity(const RouterValues& values, const manifest::Arch& arch) {
    const manifest::Router& router = *values.get_snapshot().get_router().router;
    const MultiSampledPaths paths = find_multi_sampled_paths(router, arch);
    RouterActivityVerdict activity;
    activity.loop_heartbeat = read_loop_heartbeat(
        values,
        find_single_path(paths.loop_heartbeat, manifest::MultiSampleRole::LOOP_HEARTBEAT),
        arch.heartbeat_protocol);
    activity.telemetry_heartbeats = read_telemetry_heartbeats(
        values, find_single_path(paths.fabric_telemetry, manifest::MultiSampleRole::FABRIC_TELEMETRY), router, arch);
    decide_verdict(activity, arch.heartbeat_protocol);
    return activity;
}

json make_activity_json(const RouterActivityVerdict& activity, const manifest::HeartbeatProtocol& protocol) {
    json out;
    out["loop_heartbeat"] = make_loop_heartbeat_json(activity.loop_heartbeat, protocol);
    json telemetry = json::array();
    for (const TelemetryHeartbeat& heartbeat : activity.telemetry_heartbeats) {
        telemetry.push_back(make_telemetry_heartbeat_json(heartbeat));
    }
    out["telemetry_heartbeats"] = std::move(telemetry);
    out["verdict"] = manifest::lower_enum_name(activity.verdict);
    json stopped = json::array();
    for (const auto& [erisc, side] : activity.stopped) {
        stopped.push_back({{"erisc", erisc}, {"side", manifest::lower_enum_name(side)}});
    }
    out["stopped"] = std::move(stopped);
    return out;
}

}  // namespace tt::tt_fabric::decode
