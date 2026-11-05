// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_manifest.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <span>
#include <string_view>
#include <utility>

#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_json.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_utils.hpp"
#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_files.hpp"

namespace tt::tt_fabric::capture {

// Private helpers for reading individual parts of the manifest.
namespace {

using json = JsonReader::json;
using json_pointer = JsonReader::json_pointer;

// Every stream register under `node`, by address.
void collect_stream_register_addresses(
    const JsonReader& manifest_reader,
    const json& node,
    const json_pointer& pointer,
    std::vector<uint64_t>& addresses) {
    if (node.is_object() && node.contains("stream_id")) {
        // A stream register holds stream_id, address, value_bits and type. One missing a key is a writer change
        // capture hasn't followed, so it fails rather than being skipped.
        for (const char* key : {"address", "value_bits", "type"}) {
            TT_FATAL(
                node.contains(key),
                "Fabric capture: the manifest's stream register {} has no {}",
                pointer.to_string(),
                key);
        }
        addresses.push_back(manifest_reader.read_unsigned(pointer / "address"));
        return;
    }
    if (node.is_object() || node.is_array()) {
        for (const auto& [key, child] : node.items()) {
            collect_stream_register_addresses(manifest_reader, child, pointer / key, addresses);
        }
    }
}

PlannedRouter collect_planned_router(
    const JsonReader& manifest_reader,
    const json_pointer& pointer,
    const std::string& chip_path,
    const std::string& router_key) {
    std::vector<uint64_t> addresses;
    collect_stream_register_addresses(manifest_reader, manifest_reader.read_object(pointer), pointer, addresses);
    std::ranges::sort(addresses);
    addresses.erase(std::unique(addresses.begin(), addresses.end()), addresses.end());
    return PlannedRouter{
        .path = chip_path + "/" + router_key,
        .eth_chan = manifest_reader.read_uint32(pointer / "identity" / "eth_chan"),
        .virtual_core = manifest_reader.read_core(pointer / "identity" / "virtual_core"),
        .stream_register_addresses = std::move(addresses),
    };
}

std::vector<PlannedChip> collect_planned_chips(const JsonReader& manifest_reader) {
    std::vector<PlannedChip> chips;
    const json_pointer meshes_pointer("/meshes");
    for (const auto& [mesh_key, mesh] : manifest_reader.read_object(meshes_pointer).items()) {
        const json_pointer chips_pointer = meshes_pointer / mesh_key / "chips";
        for (const auto& [chip_key, chip] : manifest_reader.read_object(chips_pointer).items()) {
            const json_pointer chip_pointer = chips_pointer / chip_key;
            if (!manifest_reader.read_bool(chip_pointer / "is_local")) {
                continue;
            }
            PlannedChip planned_chip{
                .path = mesh_key + "/" + chip_key,
                .asic_id = manifest_reader.read_asic_id(chip_pointer / "asic_id"),
            };
            const json_pointer routers_pointer = chip_pointer / "routers";
            for (const auto& [router_key, router] : manifest_reader.read_object(routers_pointer).items()) {
                planned_chip.routers.push_back(collect_planned_router(
                    manifest_reader, routers_pointer / router_key, planned_chip.path, router_key));
            }
            chips.push_back(std::move(planned_chip));
        }
    }
    return chips;
}

std::vector<CoreRegister> collect_core_registers(const JsonReader& manifest_reader) {
    const json_pointer pointer("/arch/core_registers");
    const size_t num_registers = manifest_reader.read_array(pointer).size();
    std::vector<CoreRegister> registers;
    registers.reserve(num_registers);
    for (size_t i = 0; i < num_registers; ++i) {
        registers.push_back(CoreRegister{
            .name = manifest_reader.read_string(pointer / i / "name"),
            .address = manifest_reader.read_unsigned(pointer / i / "address"),
        });
    }
    return registers;
}

// Fails on a manifest of another kind or version, whose keys may not mean what capture reads them as.
void check_manifest_kind_and_version(const JsonReader& manifest_reader) {
    const std::string file_kind = manifest_reader.read_string(json_pointer("/file_kind"));
    TT_FATAL(
        file_kind == visualizer::k_manifest_file_kind,
        "Fabric capture: the manifest's /file_kind is \"{}\", not \"{}\"",
        file_kind,
        visualizer::k_manifest_file_kind);
    const uint64_t version = manifest_reader.read_unsigned(json_pointer("/manifest_version"));
    TT_FATAL(
        version == visualizer::k_manifest_version,
        "Fabric capture: the manifest's /manifest_version is {}, but capture reads version {}",
        version,
        visualizer::k_manifest_version);
}

}  // namespace

ManifestFile read_manifest_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    TT_FATAL(in.is_open(), "Fabric capture: can't open the manifest {}", path.string());
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    TT_FATAL(!in.bad(), "Fabric capture: can't read the manifest {}", path.string());
    json parsed;
    try {
        parsed = json::parse(bytes);
    } catch (const json::parse_error& error) {
        TT_THROW("Fabric capture: the manifest {} isn't JSON: {}", path.string(), error.what());
    }
    check_manifest_kind_and_version(JsonReader(parsed, "manifest"));
    return ManifestFile{
        .path = path,
        .blake3 = compute_blake3_hex(std::as_bytes(std::span(bytes))),
        .json = std::move(parsed),
    };
}

ManifestIdentity collect_manifest_identity(const ManifestFile& manifest) {
    const JsonReader manifest_reader(manifest.json, "manifest");
    return ManifestIdentity{
        .path = manifest.path,
        .blake3 = manifest.blake3,
        .mpi_rank = manifest_reader.read_uint32(json_pointer("/run/mpi_rank")),
        .world_size = manifest_reader.read_uint32(json_pointer("/run/world_size")),
        .host_name = manifest_reader.read_string(json_pointer("/run/host_name")),
        .git_sha = manifest_reader.read_string(json_pointer("/run/build/git_sha")),
    };
}

CapturePlan collect_capture_plan(const json& manifest) {
    const JsonReader manifest_reader(manifest, "manifest");
    return CapturePlan{
        .arch = manifest_reader.read_string(json_pointer("/arch/name")),
        .l1_address = manifest_reader.read_unsigned(json_pointer("/arch/l1/address")),
        .l1_size_bytes = manifest_reader.read_uint32(json_pointer("/arch/l1/size_bytes")),
        .heartbeat_address = manifest_reader.read_unsigned(json_pointer("/arch/l1_areas/heartbeat/address")),
        .core_registers = collect_core_registers(manifest_reader),
        .chips = collect_planned_chips(manifest_reader),
    };
}

}  // namespace tt::tt_fabric::capture
