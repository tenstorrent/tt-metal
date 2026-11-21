// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One rank decoded: every router's regions read, interpreted and analysed, the rank's findings, and where it all came
// from. make_decoded_json turns it into the decoded file: the rank's manifest, written by the manifest's own writer
// with its fields per router, with each region's `decoded` and each router's `analysis` placed on its node.

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_record.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_activity.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_findings.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_inputs.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_interpret.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_router_analysis.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_slot_states.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_stream_registers.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_routing.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_fabric::decode {

// The build decode ran from, and when. The tool fills it, as capture's tool gives capture its commit, so the core
// links no build identity.
struct DecodeIdentity {
    std::string git_sha;
    bool git_dirty = false;
    std::string version;
    std::string build_type;
    // When decode ran, in UTC, as ISO 8601.
    std::string decoded_at;
};

// Where the decoded file's values came from: the inputs with their hashes, the capture, and decode itself.
struct DecodeProvenance {
    std::string manifest_file;
    std::string manifest_blake3;
    std::string capture_record_file;
    std::string capture_record_blake3;
    std::string bin_file;
    uint64_t bin_size_bytes = 0;
    std::string bin_blake3;
    std::string capture_id;
    std::string capture_git_sha;
    std::optional<std::string> cluster_id;
    std::string capture_started_at;
    std::string capture_finished_at;
    capture::RoundOptions round_options;
    std::vector<capture::DiscoverySkippedCore> discovery_skipped_cores;
    DecodeIdentity decode;
};

// One region's `decoded` (make_decoded_region_json), and where it goes below its router's node.
struct DecodedRegion {
    // for_each_router_region's path, such as "fields/erisc0/edm_status".
    std::string path;
    json decoded;
};

// One arch area's `decoded` on one router core. The decoded file gives each router an l1_areas node, at the arch
// block's keys, for it.
struct DecodedArchArea {
    // for_each_arch_area's name, such as "heartbeat".
    std::string name;
    json decoded;
};

// One router, decoded.
struct DecodedRouter {
    manifest::RouterRef ref;
    // Such as "M0/C7/E0".
    std::string path;
    // The router's manifest entry, in the facts the rank was decoded from.
    const manifest::Router* router = nullptr;
    // In for_each_router_region's order.
    std::vector<DecodedRegion> regions;
    // In for_each_arch_area's order.
    std::vector<DecodedArchArea> arch_areas;
    CaptureCoverage capture;
    // Null when capture read nothing of the router. A router whose full read failed has one from its rounds.
    std::optional<RouterActivityVerdict> activity;
    // Null in 1D.
    std::optional<std::vector<VcDirections>> directions;
    // The index of the copy of its chip's routing table (in DecodedRank::routing) that holds its own. Null when capture
    // has no full read of the router's routing table.
    std::optional<uint32_t> routing_table_copy;
    // Its count registers' full reads, for analysis across the chip's routers.
    StreamCountReads count_reads;
    // Each sender channel's uncredited_at_peer, indexed [vc][channel] as the manifest's senders are. Null where the
    // channel isn't fed by a sibling, or its or the sibling's register has no full read in range.
    std::vector<std::vector<std::optional<UncreditedAtPeer>>> uncredited_at_peer;
    // Each channel's slot states, from the channel progress log. Null when the router was built with the log off.
    std::optional<RouterSlots> slots;
};

struct DecodedRank {
    DecodeProvenance provenance;
    // In the manifest's order.
    std::vector<DecodedRouter> routers;
    // Each chip's routing table copies, in the order the chips' first routers appear.
    std::vector<ChipRouting> routing;
    std::vector<Finding> findings;

    // Whether every router was read in full, every region decoded, and nothing found that's an error: the tool's
    // exit status 0.
    bool is_complete() const;
};

// Decodes one rank from its inputs: pairs each router with its record entry, reads its regions into their types,
// interprets each into its `decoded`, and works out its analysis and findings. `request` gives the inputs' file
// names for the provenance. Reads no file but the capture record, to hash it.
DecodedRank decode_rank(const DecodeRequest& request, const DecodeInputs& inputs, DecodeIdentity identity);

// The decoded file's JSON: the decoded version and kind, the provenance and coverage, then the manifest as
// make_manifest_json writes it with FieldsLayout::PER_ROUTER, each region's `decoded` at its path below its router's
// node, each router's l1_areas and analysis, then the top-level analysis (the vocabulary and the findings).
json make_decoded_json(const DecodedRank& rank, const manifest::ManifestFacts& facts);

// Writes `decoded` to `path`, through a temporary file beside it. Without `replace`, fails if `path` already exists.
void write_decoded_file(const std::filesystem::path& path, const json& decoded, bool replace);

}  // namespace tt::tt_fabric::decode
