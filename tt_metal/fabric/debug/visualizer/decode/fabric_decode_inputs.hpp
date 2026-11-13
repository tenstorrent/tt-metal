// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_load.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_record.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_fabric::decode {

// What decode is handed initially, to find its inputs.
struct DecodeRequest {
    // The capture record. Its .bin is beside it, under the name the record gives.
    std::filesystem::path capture_record_path;
    // The manifest's path. Null takes the manifest the capture record names, which is the one capture read, in the
    // run's folder the record is in.
    std::optional<std::filesystem::path> manifest_path;
};

// One rank's inputs, read into their models and checked against each other.
struct DecodeInputs {
    // The manifest's path, the BLAKE3 of its bytes, and its model.
    capture::ManifestFile manifest;
    // The capture record, and its .bin, mapped.
    capture::LoadedCapture capture;
};

// Reads the capture, and the manifest the request gives or else the one the capture record names, and checks that the
// capture was taken from that manifest. Fails, before decode reads any value, on an input it can't use:
// - a manifest that parse_manifest_json refuses, including one from another build than decode's;
// - a capture-record that parse_capture_record refuses, or a .bin that isn't the one the record names;
// - a capture that mentions another manifest, identified by comparing the computed BLAKE3 of the manifest's bytes with
//   the one the capture mentions.
DecodeInputs load_decode_inputs(const DecodeRequest& request);

// A router in the manifest, with what capture read of it. The pointers are into the DecodeInputs the router was paired
// from.
struct CapturedRouter {
    manifest::RouterRef ref;
    // The router's path, such as "M0/C7/E0", which pairs it with its record entry.
    std::string path;
    const manifest::Chip* chip = nullptr;
    const manifest::Router* router = nullptr;
    // The record's entry for the router's chip. Null when the record has no entry for the chip.
    const capture::ChipRecord* chip_record = nullptr;
    // The record's entry for the router. Null when the record has no entry for it, so nothing of it was read.
    const capture::RouterRecord* record = nullptr;
};

// Every manifest router, each paired with its record entry, and the record entries no manifest router pairs with.
struct RouterPairing {
    // In the manifest's order. Only local chips have routers, so remote chips have none here.
    std::vector<CapturedRouter> routers;
    // The paths of the record's routers that the manifest has no router for, in the record's order. Decode skips
    // them. Capture records only the chips the manifest says are local, and the manifest gives remote chips no
    // routers, so this is empty unless the record and manifest disagree.
    std::vector<std::string> record_routers_not_in_manifest;
};

// Pairs each manifest router with the record's entry for it. Fails on a record that names a router twice.
RouterPairing pair_routers_with_record(const DecodeInputs& inputs);

}  // namespace tt::tt_fabric::decode
