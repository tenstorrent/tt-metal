// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_inputs.hpp"

#include <filesystem>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_files.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_paths.hpp"

namespace tt::tt_fabric::decode {

namespace {

// Fails unless `capture` was taken from `manifest`.
void check_capture_is_from_manifest(const capture::ManifestFile& manifest, const capture::LoadedCapture& capture) {
    const capture::ManifestIdentity& captured_from = capture.get_record().manifest;
    TT_FATAL(
        captured_from.blake3 == manifest.blake3,
        "Fabric decode: the capture was taken from a manifest with BLAKE3 {} (\"{}\"), but {} has BLAKE3 {}; decode "
        "needs the manifest the capture read",
        captured_from.blake3,
        captured_from.path.string(),
        manifest.path.string(),
        manifest.blake3);
}

// A router's entry in the record, and its chip's.
struct RouterRecordEntry {
    const capture::ChipRecord* chip_record = nullptr;
    const capture::RouterRecord* record = nullptr;
};

// The record's chip and router entries, each looked up by its path. The keys point into the record.
struct RecordEntriesByPath {
    std::unordered_map<std::string_view, const capture::ChipRecord*> chips;
    std::unordered_map<std::string_view, RouterRecordEntry> routers;
};

RecordEntriesByPath collect_record_entries_by_path(const capture::CaptureRecord& record) {
    RecordEntriesByPath entries;
    for (const capture::ChipRecord& chip : record.chips) {
        TT_FATAL(
            entries.chips.emplace(chip.path, &chip).second,
            "Fabric decode: the capture record has chip {} twice",
            chip.path);
        for (const capture::RouterRecord& router : chip.routers) {
            const std::string& path = router.target.router.path;
            TT_FATAL(
                entries.routers.emplace(path, RouterRecordEntry{.chip_record = &chip, .record = &router}).second,
                "Fabric decode: the capture record has router {} twice",
                path);
        }
    }
    return entries;
}

}  // namespace

DecodeInputs load_decode_inputs(const DecodeRequest& request) {
    // Read the capture record, and the manifest the request gives, else the one the record names. The record names it
    // by its file name in the run's folder, which the record is in.
    capture::LoadedCapture capture = capture::load_capture(request.capture_record_path);
    const std::filesystem::path manifest_path = request.manifest_path.value_or(
        visualizer::get_run_directory_of_capture_record(request.capture_record_path) /
        capture.get_record().manifest.path);
    TT_FATAL(
        request.manifest_path.has_value() || std::filesystem::exists(manifest_path),
        "Fabric decode: {} was captured from {}, which isn't there; give the manifest with --manifest",
        request.capture_record_path.string(),
        manifest_path.string());
    capture::ManifestFile manifest = capture::read_manifest_file(manifest_path);

    // Validate that the capture was taken from the manifest
    check_capture_is_from_manifest(manifest, capture);

    return DecodeInputs{.manifest = std::move(manifest), .capture = std::move(capture)};
}

RouterPairing pair_routers_with_record(const DecodeInputs& inputs) {
    const capture::CaptureRecord& record = inputs.capture.get_record();
    RecordEntriesByPath entries = collect_record_entries_by_path(record);
    RouterPairing pairing;

    // Go through every router in every chip in every mesh listed in the manifest, and pair it with the router's
    // corresponding record from the capture record
    for (const manifest::Mesh& mesh : inputs.manifest.facts.meshes) {
        for (const manifest::Chip& chip : mesh.chips) {
            const auto chip_record = entries.chips.find(manifest::chip_path(chip.fabric_node_id));
            for (const manifest::Router& router : chip.routers) {
                CapturedRouter captured{
                    .ref = {.node = chip.fabric_node_id, .key = router.identity.key},
                    .chip = &chip,
                    .router = &router,
                    .chip_record = chip_record == entries.chips.end() ? nullptr : chip_record->second,
                };
                captured.path = manifest::router_path(captured.ref);
                // Paired entries are removed, so what's left at the end is what the manifest doesn't have.
                if (const auto entry = entries.routers.find(captured.path); entry != entries.routers.end()) {
                    captured.chip_record = entry->second.chip_record;
                    captured.record = entry->second.record;
                    entries.routers.erase(entry);
                }
                pairing.routers.push_back(std::move(captured));
            }
        }
    }

    // Mark the routers that were not found in the manifest
    for (const capture::ChipRecord& chip : record.chips) {
        for (const capture::RouterRecord& router : chip.routers) {
            if (entries.routers.contains(router.target.router.path)) {
                pairing.record_routers_not_in_manifest.push_back(router.target.router.path);
            }
        }
    }
    return pairing;
}

}  // namespace tt::tt_fabric::decode
