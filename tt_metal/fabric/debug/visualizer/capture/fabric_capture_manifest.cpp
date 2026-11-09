// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_manifest.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <utility>
#include <variant>

#include <nlohmann/json.hpp>
#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_utils.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_paths.hpp"

namespace tt::tt_fabric::capture {

// Private helpers for reading individual parts of the manifest.
namespace {

// Adds the address of every stream register `fields_by_erisc` names.
void collect_stream_register_addresses(
    const manifest::FieldsByErisc& fields_by_erisc, std::vector<uint64_t>& addresses) {
    for (const std::vector<manifest::Field>& fields : fields_by_erisc) {
        for (const manifest::Field& field : fields) {
            if (!field.content.has_value()) {
                continue;
            }
            if (const auto* stream = std::get_if<manifest::content::StreamReg>(&*field.content)) {
                addresses.push_back(stream->address);
            }
        }
    }
}

// Every stream register a router names, ascending, each once. Stream registers are only ever field contents: the
// router's own fields, its channels' and its edges'.
std::vector<uint64_t> collect_router_stream_register_addresses(const manifest::Router& router) {
    std::vector<uint64_t> addresses;
    collect_stream_register_addresses(router.fields_by_erisc, addresses);
    for (const auto& senders : router.channels.senders) {
        for (const manifest::SenderChannel& sender : senders) {
            collect_stream_register_addresses(sender.fields_by_erisc, addresses);
        }
    }
    for (const auto& receivers : router.channels.receivers) {
        for (const manifest::ReceiverChannel& receiver : receivers) {
            collect_stream_register_addresses(receiver.fields_by_erisc, addresses);
        }
    }
    for (const auto& edges : router.intra_chip_downstream_edges) {
        for (const manifest::DownstreamEdge& edge : edges) {
            collect_stream_register_addresses(edge.fields_by_erisc, addresses);
        }
    }
    std::ranges::sort(addresses);
    addresses.erase(std::unique(addresses.begin(), addresses.end()), addresses.end());
    return addresses;
}

PlannedRouter collect_planned_router(const manifest::Chip& chip, const manifest::Router& router) {
    return PlannedRouter{
        .path = manifest::router_path({.node = chip.fabric_node_id, .key = router.identity.key}),
        .eth_chan = router.identity.eth_chan,
        .virtual_core =
            {
                .x = static_cast<uint32_t>(router.identity.virtual_core.x),
                .y = static_cast<uint32_t>(router.identity.virtual_core.y),
            },
        .stream_register_addresses = collect_router_stream_register_addresses(router),
    };
}

// The chips this rank owns: the ones with a physical chip id. The reader gives each of them an asic_id.
std::vector<PlannedChip> collect_planned_chips(const manifest::ManifestFacts& facts) {
    std::vector<PlannedChip> chips;
    for (const manifest::Mesh& mesh : facts.meshes) {
        for (const manifest::Chip& chip : mesh.chips) {
            if (!chip.physical_chip_id.has_value()) {
                continue;
            }
            PlannedChip planned_chip{
                .path = manifest::mesh_key(chip.fabric_node_id.mesh_id) + "/" +
                        manifest::chip_key(chip.fabric_node_id.chip_id),
                .asic_id = chip.asic_id.value(),
            };
            for (const manifest::Router& router : chip.routers) {
                planned_chip.routers.push_back(collect_planned_router(chip, router));
            }
            chips.push_back(std::move(planned_chip));
        }
    }
    return chips;
}

std::vector<CoreRegister> collect_core_registers(const manifest::CoreRegisters& registers) {
    std::vector<CoreRegister> out;
    out.reserve(manifest::k_core_registers.size());
    for (const auto& [name, address] : manifest::k_core_registers) {
        out.push_back(CoreRegister{.name = std::string(name), .address = registers.*address});
    }
    return out;
}

}  // namespace

ManifestFile read_manifest_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    TT_FATAL(in.is_open(), "Fabric capture: can't open the manifest {}", path.string());
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    TT_FATAL(!in.bad(), "Fabric capture: can't read the manifest {}", path.string());
    nlohmann::ordered_json parsed;
    try {
        parsed = nlohmann::ordered_json::parse(bytes);
    } catch (const nlohmann::ordered_json::parse_error& error) {
        TT_THROW("Fabric capture: the manifest {} isn't JSON: {}", path.string(), error.what());
    }
    return ManifestFile{
        .path = path,
        .blake3 = compute_blake3_hex(std::as_bytes(std::span(bytes))),
        .facts = parse_manifest_json(parsed),
    };
}

ManifestIdentity collect_manifest_identity(const ManifestFile& manifest_file) {
    const manifest::RunInfo& run = manifest_file.facts.run;
    return ManifestIdentity{
        .path = manifest_file.path,
        .blake3 = manifest_file.blake3,
        .run_id = run.run_id,
        .mpi_rank = run.mpi_rank,
        .world_size = run.world_size,
        .host_name = run.host_name,
        .git_sha = std::string(run.build.git_sha),
    };
}

CapturePlan collect_capture_plan(const manifest::ManifestFacts& facts) {
    return CapturePlan{
        .arch = facts.arch.arch,
        .l1_address = facts.arch.l1.address,
        .l1_size_bytes = facts.arch.l1.size_bytes,
        .heartbeat_address = facts.arch.areas.heartbeat.word.address,
        .core_registers = collect_core_registers(facts.arch.core_registers),
        .chips = collect_planned_chips(facts),
    };
}

}  // namespace tt::tt_fabric::capture
