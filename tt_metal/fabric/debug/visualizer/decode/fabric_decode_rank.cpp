// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_rank.hpp"

#include <algorithm>
#include <fstream>
#include <ios>
#include <iterator>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_utils.hpp"
#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_files.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_packet_header.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_packet_path.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_region_json.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_routing.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_routing_table.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_snapshot.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_stream_registers.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_values.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_paths.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_regions.hpp"

namespace tt::tt_fabric::decode {

namespace {

// The BLAKE3 of the file at `path`'s bytes.
std::string hash_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    TT_FATAL(in.is_open(), "Fabric decode: can't open {} to hash it", path.string());
    const std::vector<char> chars{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    std::vector<std::byte> bytes(chars.size());
    std::ranges::transform(chars, bytes.begin(), [](char c) { return static_cast<std::byte>(c); });
    return capture::compute_blake3_hex(bytes);
}

DecodeProvenance make_provenance(const DecodeRequest& request, const DecodeInputs& inputs, DecodeIdentity identity) {
    const capture::CaptureRecord& record = inputs.capture.get_record();
    return DecodeProvenance{
        .manifest_file = inputs.manifest.path.filename().string(),
        .manifest_blake3 = inputs.manifest.blake3,
        .capture_record_file = request.capture_record_path.filename().string(),
        .capture_record_blake3 = hash_file(request.capture_record_path),
        .bin_file = record.bin_file,
        .bin_size_bytes = record.bin_size_bytes,
        .bin_blake3 = record.bin_blake3,
        .capture_id = record.capture_id,
        .capture_git_sha = record.capture_git_sha,
        .cluster_id = record.cluster_id,
        .capture_started_at = record.started_at,
        .capture_finished_at = record.finished_at,
        .round_options = record.round_options,
        .discovery_skipped_cores = record.discovery_skipped_cores,
        .decode = std::move(identity),
    };
}

// One router, from its snapshot: every region's and arch area's `decoded`, its analysis, and its findings.
// Its routing table's full read, when it has one, is added to `routing_tables`, for analyze_routers_routing_tables.
DecodedRouter decode_router(
    const RouterSnapshot& snapshot,
    const manifest::ManifestFacts& facts,
    const RouteView& routes,
    std::vector<Finding>& findings,
    std::vector<RouterRoutingTable>& routing_tables) {
    const manifest::Arch& arch = facts.arch;
    const RouterValues values(snapshot, arch);
    const CapturedRouter& captured = snapshot.get_router();
    const manifest::Router& router = *captured.router;
    DecodedRouter decoded{.ref = captured.ref, .path = captured.path, .router = captured.router};
    const StreamRegisterFactsByRegister register_facts = collect_stream_register_facts(facts, captured.ref, router);
    manifest::for_each_router_region(router, [&](std::string_view path, const auto& region) {
        const InterpretContext context = make_interpret_context(facts, routes, captured.ref, router, path);
        json region_decoded;
        if constexpr (std::is_same_v<std::decay_t<decltype(region)>, manifest::content::StreamReg>) {
            region_decoded = make_decoded_region_json(values, region, register_facts.at(&region), context);
        } else {
            region_decoded = make_decoded_region_json(values, region, context);
        }
        decoded.regions.push_back(DecodedRegion{.path = std::string(path), .decoded = std::move(region_decoded)});
    });
    manifest::for_each_arch_area(arch, [&](std::string_view name, const manifest::content::L1& area) {
        const std::string path = make_arch_area_path(name);
        const InterpretContext context = make_interpret_context(facts, routes, captured.ref, router, path);
        decoded.arch_areas.push_back(
            DecodedArchArea{.name = std::string(name), .decoded = make_decoded_region_json(values, area, context)});
    });
    decoded.capture = summarize_capture(values);
    if (decoded.capture.status != RouterReadStatus::NOT_READ) {
        decoded.activity = judge_router_activity(values, arch);
    }
    decoded.directions = find_router_directions(router);
    decoded.count_reads = collect_stream_count_reads(values);
    if (std::optional<RoutingTableRead> read = find_routing_table_read(values)) {
        routing_tables.push_back(RouterRoutingTable{
            .ref = captured.ref, .path = captured.path, .router = captured.router, .read = std::move(*read)});
    }
    std::ranges::move(
        find_router_findings(values, arch, register_facts, decoded.directions), std::back_inserter(findings));
    return decoded;
}

// Every value of an enum the decoded file writes, by lower_enum_name, for analysis.vocabulary.
template <typename E>
json make_vocabulary_list_json() {
    json names = json::array();
    for (const E value : enchantum::values<E>) {
        names.push_back(manifest::lower_enum_name(value));
    }
    return names;
}

json make_inputs_json(const DecodeProvenance& provenance) {
    json out;
    out["manifest"] = {{"file", provenance.manifest_file}, {"blake3", provenance.manifest_blake3}};
    out["capture_record"] = {{"file", provenance.capture_record_file}, {"blake3", provenance.capture_record_blake3}};
    out["bin"] = {
        {"file", provenance.bin_file}, {"size_bytes", provenance.bin_size_bytes}, {"blake3", provenance.bin_blake3}};
    return out;
}

json make_capture_provenance_json(const DecodeProvenance& provenance) {
    json out;
    out["capture_id"] = provenance.capture_id;
    out["git_sha"] = provenance.capture_git_sha;
    out["cluster_id"] = provenance.cluster_id.has_value() ? json(*provenance.cluster_id) : json(nullptr);
    out["started_at"] = provenance.capture_started_at;
    out["finished_at"] = provenance.capture_finished_at;
    out["rounds"] = {
        {"num_rounds", provenance.round_options.num_rounds},
        {"interval_ms", static_cast<uint64_t>(provenance.round_options.interval.count())}};
    json skipped = json::array();
    for (const capture::DiscoverySkippedCore& core : provenance.discovery_skipped_cores) {
        skipped.push_back(
            {{"asic_id", manifest::make_asic_id_text(core.asic_id)},
             {"core", {{"x", core.core.x}, {"y", core.core.y}}},
             {"reason", core.reason}});
    }
    out["discovery_skipped_cores"] = std::move(skipped);
    return out;
}

json make_decode_identity_json(const DecodeIdentity& identity) {
    json out;
    out["git_sha"] = identity.git_sha;
    out["git_dirty"] = identity.git_dirty;
    out["version"] = identity.version;
    out["build_type"] = identity.build_type;
    out["decoded_at"] = identity.decoded_at;
    return out;
}

// How much of the rank decode has values for.
json make_coverage_json(const DecodedRank& rank) {
    const auto count_status = [&](RouterReadStatus status) {
        return std::ranges::count_if(
            rank.routers, [&](const DecodedRouter& router) { return router.capture.status == status; });
    };
    size_t num_regions = 0;
    size_t num_regions_unread = 0;
    for (const DecodedRouter& router : rank.routers) {
        num_regions += router.capture.num_decoded + router.capture.num_inactive + router.capture.num_unread;
        num_regions_unread += router.capture.num_unread;
    }
    json out;
    out["num_routers"] = rank.routers.size();
    out["num_routers_complete"] = count_status(RouterReadStatus::COMPLETE);
    out["num_routers_partial"] = count_status(RouterReadStatus::PARTIAL);
    out["num_routers_not_read"] = count_status(RouterReadStatus::NOT_READ);
    out["num_regions"] = num_regions;
    out["num_regions_unread"] = num_regions_unread;
    return out;
}

// Puts the router's `decoded` values, its l1_areas and its analysis on its node.
void place_router(
    const DecodedRouter& router,
    const manifest::HeartbeatProtocol& protocol,
    const json& arch_areas,
    json& manifest_json) {
    const json::json_pointer router_pointer(manifest::router_node_pointer(router.ref));
    TT_FATAL(
        manifest_json.contains(router_pointer), "Fabric decode: the manifest's JSON has no node for {}", router.path);
    json& node = manifest_json[router_pointer];
    for (const DecodedRegion& region : router.regions) {
        const json::json_pointer region_pointer("/" + region.path);
        TT_FATAL(
            node.contains(region_pointer) && node[region_pointer].is_object(),
            "Fabric decode: {} has no region node at {}, though for_each_router_region walks it",
            router.path,
            region.path);
        node[region_pointer]["decoded"] = region.decoded;
    }
    // Every router core has its own copy of the arch areas: its l1_areas node has the arch block's keys for each,
    // and its own decoded.
    json l1_areas = json::object();
    for (const DecodedArchArea& area : router.arch_areas) {
        json area_json = arch_areas.at(area.name);
        area_json["decoded"] = area.decoded;
        l1_areas[area.name] = std::move(area_json);
    }
    node["l1_areas"] = std::move(l1_areas);
    for (size_t vc = 0; vc < router.uncredited_at_peer.size(); ++vc) {
        for (size_t channel = 0; channel < router.uncredited_at_peer[vc].size(); ++channel) {
            const json::json_pointer channel_pointer(fmt::format("/channels/senders/vc{}/ch{}", vc, channel));
            TT_FATAL(
                node.contains(channel_pointer),
                "Fabric decode: {} has no sender channel node at {}",
                router.path,
                channel_pointer.to_string());
            node[channel_pointer]["analysis"]["uncredited_at_peer"] =
                make_uncredited_at_peer_json(router.uncredited_at_peer[vc][channel]);
        }
    }
    json analysis;
    analysis["capture"] = make_capture_json(router.capture);
    analysis["activity"] = router.activity.has_value() ? make_activity_json(*router.activity, protocol) : json(nullptr);
    analysis["directions"] = make_directions_json(router.directions);
    analysis["routing_table"] = make_router_routing_table_json(router.routing_table_copy);
    node["analysis"] = std::move(analysis);
}

json make_analysis_json(const DecodedRank& rank) {
    json vocabulary;
    vocabulary["finding_code"] = make_vocabulary_list_json<FindingCode>();
    vocabulary["finding_severity"] = make_vocabulary_list_json<FindingSeverity>();
    vocabulary["finding_origin"] = make_vocabulary_list_json<FindingOrigin>();
    vocabulary["router_activity"] = make_vocabulary_list_json<RouterActivity>();
    vocabulary["read_status"] = make_vocabulary_list_json<RouterReadStatus>();
    vocabulary["unread_code"] = make_vocabulary_list_json<UnreadCode>();
    vocabulary["direction_table_entry"] = make_direction_table_entry_vocabulary_json();
    vocabulary["axis_code_2d"] = make_2d_axis_code_vocabulary_json();
    vocabulary["hop_action_1d"] = make_1d_hop_action_vocabulary_json();
    vocabulary["action_2d"] = make_2d_action_vocabulary_json();
    vocabulary["path_end"] = make_vocabulary_list_json<PathEnd>();
    json findings = json::array();
    for (const Finding& finding : rank.findings) {
        findings.push_back(make_finding_json(finding));
    }
    json out;
    out["vocabulary"] = std::move(vocabulary);
    out["findings"] = std::move(findings);
    return out;
}

// Groups each chip's routers' routing tables into the rank's routing, and tells each router which copy holds its own.
void analyze_routers_routing_tables(
    DecodedRank& rank, std::span<const RouterRoutingTable> routing_tables, const manifest::ManifestFacts& facts) {
    RoutingAnalysis analysis = analyze_routing_tables(routing_tables, facts, rank.findings);
    rank.routing = std::move(analysis.chips);
    for (DecodedRouter& router : rank.routers) {
        if (const auto copy = analysis.copy_by_router.find(router.path); copy != analysis.copy_by_router.end()) {
            router.routing_table_copy = copy->second;
        }
    }
}

// The decoded router on `ref`, or null when the rank didn't decode one there.
const DecodedRouter* find_decoded_router(const DecodedRank& rank, const manifest::RouterRef& ref) {
    const auto router = std::ranges::find_if(rank.routers, [&](const DecodedRouter& decoded) {
        return decoded.ref.node == ref.node && decoded.ref.key.direction == ref.key.direction &&
               decoded.ref.key.routing_plane == ref.key.routing_plane;
    });
    return router == rank.routers.end() ? nullptr : &*router;
}

// Each router's sender channels' uncredited_at_peer, from each channel's free slots and its producer's edge into it,
// and a finding for each that's negative. A sibling is on the same chip, so the rank has its reads whenever capture
// read it.
void analyze_uncredited_at_peer(DecodedRank& rank) {
    for (DecodedRouter& consumer : rank.routers) {
        const auto& senders = consumer.router->channels.senders;
        consumer.uncredited_at_peer.resize(senders.size());
        for (size_t vc = 0; vc < senders.size(); ++vc) {
            consumer.uncredited_at_peer[vc].resize(senders[vc].size());
            for (size_t channel = 0; channel < senders[vc].size(); ++channel) {
                const std::optional<manifest::SenderChannelProducer>& producer = senders[vc][channel].producer;
                const auto* sibling =
                    producer.has_value() ? std::get_if<manifest::SiblingRouterRef>(&*producer) : nullptr;
                if (sibling == nullptr) {
                    continue;
                }
                const manifest::RouterRef producer_ref{
                    .node = consumer.ref.node,
                    .key = {.direction = sibling->direction, .routing_plane = consumer.ref.key.routing_plane}};
                const DecodedRouter* producer_router = find_decoded_router(rank, producer_ref);
                if (producer_router == nullptr) {
                    continue;
                }
                std::optional<UncreditedAtPeer> value = find_uncredited_at_peer(
                    {.router = *consumer.router, .path = consumer.path, .reads = consumer.count_reads},
                    vc,
                    channel,
                    {.router = *producer_router->router,
                     .path = producer_router->path,
                     .reads = producer_router->count_reads});
                if (value.has_value() && value->value < 0) {
                    rank.findings.push_back(Finding{
                        .code = FindingCode::UNCREDITED_AT_PEER_NEGATIVE,
                        .paths =
                            {fmt::format("{}/channels/senders/vc{}/ch{}", consumer.path, vc, channel),
                             value->producer_edge},
                        .message = fmt::format(
                            "{} uncredited at the producer, less {} unsent, is {}: the two registers were read {} ns "
                            "apart while the routers moved",
                            value->producer_uncredited,
                            value->unsent,
                            value->value,
                            value->read_apart_ns)});
                }
                consumer.uncredited_at_peer[vc][channel] = std::move(value);
            }
        }
    }
}

}  // namespace

bool DecodedRank::is_complete() const {
    const bool all_routers_complete = std::ranges::all_of(routers, [](const DecodedRouter& router) {
        return router.capture.status == RouterReadStatus::COMPLETE && router.capture.num_unread == 0;
    });
    const bool no_errors = std::ranges::none_of(
        findings, [](const Finding& finding) { return get_finding_severity(finding.code) == FindingSeverity::ERROR; });
    return all_routers_complete && no_errors;
}

DecodedRank decode_rank(const DecodeRequest& request, const DecodeInputs& inputs, DecodeIdentity identity) {
    DecodedRank rank{.provenance = make_provenance(request, inputs, std::move(identity))};
    const manifest::Arch& arch = inputs.manifest.facts.arch;
    const RouterPairing pairing = pair_routers_with_record(inputs);
    rank.findings = find_pairing_findings(pairing);
    const std::vector<RouterSnapshot> snapshots = make_router_snapshots(pairing, inputs.capture, arch.l1);
    rank.routers.reserve(snapshots.size());
    std::vector<RouterRoutingTable> routing_tables;
    const RouteView routes(inputs.manifest.facts);
    for (const RouterSnapshot& snapshot : snapshots) {
        rank.routers.push_back(decode_router(snapshot, inputs.manifest.facts, routes, rank.findings, routing_tables));
    }
    analyze_routers_routing_tables(rank, routing_tables, inputs.manifest.facts);
    analyze_uncredited_at_peer(rank);
    return rank;
}

json make_decoded_json(const DecodedRank& rank, const manifest::ManifestFacts& facts) {
    json manifest_json = make_manifest_json(facts, FieldsLayout::PER_ROUTER);
    const json arch_areas = manifest_json.at("arch").at("l1_areas");
    const RouteView routes(facts);
    for (const DecodedRouter& router : rank.routers) {
        place_router(router, facts.arch.heartbeat_protocol, arch_areas, manifest_json);
    }
    for (const ChipRouting& routing : rank.routing) {
        const json::json_pointer chip_pointer(manifest::chip_node_pointer(routing.chip));
        TT_FATAL(
            manifest_json.contains(chip_pointer),
            "Fabric decode: the manifest's JSON has no node for {}",
            manifest::chip_path(routing.chip));
        manifest_json[chip_pointer]["analysis"]["routing"] = make_chip_routing_json(routing, facts, routes);
    }

    json out;
    out["decoded_version"] = visualizer::k_decoded_version;
    out["file_kind"] = std::string(visualizer::k_decoded_file_kind);
    out["manifest_version"] = manifest_json.at("manifest_version");
    out["inputs"] = make_inputs_json(rank.provenance);
    out["capture"] = make_capture_provenance_json(rank.provenance);
    out["decode"] = make_decode_identity_json(rank.provenance.decode);
    out["coverage"] = make_coverage_json(rank);
    // The manifest within, as its writer wrote it, but for its own version and kind, which are above.
    for (auto& [key, value] : manifest_json.items()) {
        if (key != "manifest_version" && key != "file_kind") {
            out[key] = std::move(value);
        }
    }
    out["analysis"] = make_analysis_json(rank);
    return out;
}

void write_decoded_file(const std::filesystem::path& path, const json& decoded, bool replace) {
    TT_FATAL(
        replace || !std::filesystem::exists(path),
        "Fabric decode: {} already exists; pass --replace to replace it",
        path.string());
    const std::filesystem::path temporary_path = visualizer::make_temporary_path(path);
    try {
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        std::ofstream out;
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out.open(temporary_path);
        out << decoded.dump(2) << '\n';
        out.close();
        const std::error_code error = visualizer::move_into_place(temporary_path, path, replace);
        TT_FATAL(
            !error, "Fabric decode: can't move {} to {}: {}", temporary_path.string(), path.string(), error.message());
    } catch (...) {
        std::error_code error;
        std::filesystem::remove(temporary_path, error);
        throw;
    }
}

}  // namespace tt::tt_fabric::decode
