// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_manifest.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <limits>
#include <span>
#include <string_view>
#include <utility>

#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_utils.hpp"
#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_files.hpp"

namespace tt::tt_fabric::capture {

// Private helpers for reading individual parts of the manifest.
namespace {

using json = nlohmann::ordered_json;
using json_pointer = json::json_pointer;

// The node at `pointer`. Fails, naming the pointer, when the manifest has none.
const json& find_required(const json& manifest, const json_pointer& pointer) {
    TT_FATAL(manifest.contains(pointer), "Fabric capture: the manifest has no {}", pointer.to_string());
    return manifest.at(pointer);
}

uint64_t read_unsigned(const json& manifest, const json_pointer& pointer) {
    const json& value = find_required(manifest, pointer);
    TT_FATAL(
        value.is_number_unsigned(),
        "Fabric capture: the manifest's {} is {}, not an unsigned integer",
        pointer.to_string(),
        value.dump());
    return value.get<uint64_t>();
}

uint32_t read_uint32(const json& manifest, const json_pointer& pointer) {
    const uint64_t value = read_unsigned(manifest, pointer);
    TT_FATAL(
        value <= std::numeric_limits<uint32_t>::max(),
        "Fabric capture: the manifest's {} is {}, which doesn't fit 32 bits",
        pointer.to_string(),
        value);
    return static_cast<uint32_t>(value);
}

std::string read_string(const json& manifest, const json_pointer& pointer) {
    const json& value = find_required(manifest, pointer);
    TT_FATAL(
        value.is_string(), "Fabric capture: the manifest's {} is {}, not a string", pointer.to_string(), value.dump());
    return value.get<std::string>();
}

bool read_bool(const json& manifest, const json_pointer& pointer) {
    const json& value = find_required(manifest, pointer);
    TT_FATAL(
        value.is_boolean(),
        "Fabric capture: the manifest's {} is {}, not a boolean",
        pointer.to_string(),
        value.dump());
    return value.get<bool>();
}

const json& read_object(const json& manifest, const json_pointer& pointer) {
    const json& value = find_required(manifest, pointer);
    TT_FATAL(value.is_object(), "Fabric capture: the manifest's {} isn't an object", pointer.to_string());
    return value;
}

const json& read_array(const json& manifest, const json_pointer& pointer) {
    const json& value = find_required(manifest, pointer);
    TT_FATAL(value.is_array(), "Fabric capture: the manifest's {} isn't an array", pointer.to_string());
    return value;
}

// A chip's asic_id, which the manifest writes as hex, such as "0x00000000deadbeef".
uint64_t read_asic_id(const json& manifest, const json_pointer& pointer) {
    const std::string text = read_string(manifest, pointer);
    const std::string_view digits = std::string_view(text).substr(text.starts_with("0x") ? 2 : 0);
    TT_FATAL(
        text.starts_with("0x") && !digits.empty() && digits.size() <= 16 &&
            std::ranges::all_of(digits, [](char c) { return std::isxdigit(static_cast<unsigned char>(c)) != 0; }),
        "Fabric capture: the manifest's {} is \"{}\", not a hex asic id",
        pointer.to_string(),
        text);
    return std::stoull(std::string(digits), nullptr, 16);
}

// A core's coordinates, which the manifest writes as [x, y].
CoreXY read_core(const json& manifest, const json_pointer& pointer) {
    const json& core = read_array(manifest, pointer);
    TT_FATAL(core.size() == 2, "Fabric capture: the manifest's {} isn't an [x, y] pair", pointer.to_string());
    return CoreXY{.x = read_uint32(manifest, pointer / 0), .y = read_uint32(manifest, pointer / 1)};
}

// Every stream register under `node`, by address.
void collect_stream_register_addresses(
    const json& manifest, const json& node, const json_pointer& pointer, std::vector<uint64_t>& addresses) {
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
        addresses.push_back(read_unsigned(manifest, pointer / "address"));
        return;
    }
    if (node.is_object() || node.is_array()) {
        for (const auto& [key, child] : node.items()) {
            collect_stream_register_addresses(manifest, child, pointer / key, addresses);
        }
    }
}

PlannedRouter collect_planned_router(
    const json& manifest, const json_pointer& pointer, const std::string& chip_path, const std::string& router_key) {
    std::vector<uint64_t> addresses;
    collect_stream_register_addresses(manifest, read_object(manifest, pointer), pointer, addresses);
    std::ranges::sort(addresses);
    addresses.erase(std::unique(addresses.begin(), addresses.end()), addresses.end());
    return PlannedRouter{
        .path = chip_path + "/" + router_key,
        .eth_chan = read_uint32(manifest, pointer / "identity" / "eth_chan"),
        .virtual_core = read_core(manifest, pointer / "identity" / "virtual_core"),
        .stream_register_addresses = std::move(addresses),
    };
}

std::vector<PlannedChip> collect_planned_chips(const json& manifest) {
    std::vector<PlannedChip> chips;
    const json_pointer meshes_pointer("/meshes");
    for (const auto& [mesh_key, mesh] : read_object(manifest, meshes_pointer).items()) {
        const json_pointer chips_pointer = meshes_pointer / mesh_key / "chips";
        for (const auto& [chip_key, chip] : read_object(manifest, chips_pointer).items()) {
            const json_pointer chip_pointer = chips_pointer / chip_key;
            if (!read_bool(manifest, chip_pointer / "is_local")) {
                continue;
            }
            PlannedChip planned_chip{
                .path = mesh_key + "/" + chip_key,
                .asic_id = read_asic_id(manifest, chip_pointer / "asic_id"),
            };
            const json_pointer routers_pointer = chip_pointer / "routers";
            for (const auto& [router_key, router] : read_object(manifest, routers_pointer).items()) {
                planned_chip.routers.push_back(
                    collect_planned_router(manifest, routers_pointer / router_key, planned_chip.path, router_key));
            }
            chips.push_back(std::move(planned_chip));
        }
    }
    return chips;
}

std::vector<CoreRegister> collect_core_registers(const json& manifest) {
    const json_pointer pointer("/arch/core_registers");
    const size_t num_registers = read_array(manifest, pointer).size();
    std::vector<CoreRegister> registers;
    registers.reserve(num_registers);
    for (size_t i = 0; i < num_registers; ++i) {
        registers.push_back(CoreRegister{
            .name = read_string(manifest, pointer / i / "name"),
            .address = read_unsigned(manifest, pointer / i / "address"),
        });
    }
    return registers;
}

// Fails on a manifest of another kind or version, whose keys may not mean what capture reads them as.
void check_manifest_kind_and_version(const json& manifest) {
    const std::string file_kind = read_string(manifest, json_pointer("/file_kind"));
    TT_FATAL(
        file_kind == visualizer::k_manifest_file_kind,
        "Fabric capture: the manifest's /file_kind is \"{}\", not \"{}\"",
        file_kind,
        visualizer::k_manifest_file_kind);
    const uint64_t version = read_unsigned(manifest, json_pointer("/manifest_version"));
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
    return ManifestFile{
        .path = path,
        .blake3 = compute_blake3_hex(std::as_bytes(std::span(bytes))),
        .json = std::move(parsed),
    };
}

ManifestIdentity collect_manifest_identity(const ManifestFile& manifest) {
    check_manifest_kind_and_version(manifest.json);
    return ManifestIdentity{
        .path = manifest.path,
        .blake3 = manifest.blake3,
        .mpi_rank = read_uint32(manifest.json, json_pointer("/run/mpi_rank")),
        .world_size = read_uint32(manifest.json, json_pointer("/run/world_size")),
        .host_name = read_string(manifest.json, json_pointer("/run/host_name")),
        .git_sha = read_string(manifest.json, json_pointer("/run/build/git_sha")),
    };
}

CapturePlan collect_capture_plan(const json& manifest) {
    check_manifest_kind_and_version(manifest);
    return CapturePlan{
        .arch = read_string(manifest, json_pointer("/arch/name")),
        .l1_address = read_unsigned(manifest, json_pointer("/arch/l1/address")),
        .l1_size_bytes = read_uint32(manifest, json_pointer("/arch/l1/size_bytes")),
        .heartbeat_address = read_unsigned(manifest, json_pointer("/arch/l1_areas/heartbeat/address")),
        .core_registers = collect_core_registers(manifest),
        .chips = collect_planned_chips(manifest),
    };
}

}  // namespace tt::tt_fabric::capture
