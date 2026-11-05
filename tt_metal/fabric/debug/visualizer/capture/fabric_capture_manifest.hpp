// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_utils.hpp"

namespace tt::tt_fabric::capture {

// The manifest file a capture reads. We keep the JSON representation of the manifest in memory only to collect the plan
// and identity. Afterwards, we let it go since we have taken all the information we need from it.
struct ManifestFile {
    // Path to manifest file
    std::filesystem::path path;
    // Hash of manifest file's bytes
    std::string blake3;
    // Parsed manifest file in json
    nlohmann::ordered_json json;
};

// Reads a manifest file, hashing and parsing the same bytes.
ManifestFile read_manifest_file(const std::filesystem::path& path);

// Which run a manifest is from: what a capture records about the manifest it read, and names its files by.
struct ManifestIdentity {
    std::filesystem::path path;
    std::string blake3;
    // The rank in the whole job that wrote the manifest, and the job's number of ranks.
    uint32_t mpi_rank = 0;
    uint32_t world_size = 1;
    std::string host_name;
    std::string git_sha;

    bool operator==(const ManifestIdentity&) const = default;
};

// Collects which run a manifest is from.
ManifestIdentity collect_manifest_identity(const ManifestFile& manifest);

// A register every router core has, such as the wall clock.
struct CoreRegister {
    std::string name;
    uint64_t address = 0;

    bool operator==(const CoreRegister&) const = default;
};

// PlannedChip and PlannedRouter are the chips and routers that we get from the manifest. They are "planned" because we
// need to validate that they exist in the physical cluster we are on, which we do by checking against ClusterFacts.

// A router on a chip this rank owns.
struct PlannedRouter {
    // The router's path, such as "M0/C7/E0".
    std::string path;
    uint32_t eth_chan = 0;
    CoreXY virtual_core;
    // Every stream register the manifest names for the router, by address: ascending, each once.
    std::vector<uint64_t> stream_register_addresses;

    bool operator==(const PlannedRouter&) const = default;
};

// A chip this rank owns. Contains its routers.
struct PlannedChip {
    // The chip's path, such as "M0/C7".
    std::string path;
    uint64_t asic_id = 0;
    std::vector<PlannedRouter> routers;

    bool operator==(const PlannedChip&) const = default;
};

// What a capture reads, from one rank's manifest: which cores, and where to read from in each core.
struct CapturePlan {
    // The architecture the addresses below are for. A chip of another architecture isn't read.
    std::string arch;
    // The L1 of the Ethernet core a router runs on: l1_size_bytes bytes from l1_address.
    uint64_t l1_address = 0;
    uint32_t l1_size_bytes = 0;
    uint64_t heartbeat_address = 0;
    // In the manifest's order, which is the order they're read in: the low wall clock word latches the high one.
    std::vector<CoreRegister> core_registers;
    // The chips this rank owns, in the manifest's order. Other ranks' chips are left out.
    std::vector<PlannedChip> chips;

    bool operator==(const CapturePlan&) const = default;
};

// Collects what a capture needs to read from hardware using the provided manifest. Fails on a manifest of another kind
// or version, or one missing a key the plan needs, naming the key's JSON pointer.
CapturePlan collect_capture_plan(const nlohmann::ordered_json& manifest);

}  // namespace tt::tt_fabric::capture
