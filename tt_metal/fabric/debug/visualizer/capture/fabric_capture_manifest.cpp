// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_manifest.hpp"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iterator>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_utils.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_paths.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_regions.hpp"

namespace tt::tt_fabric::capture {

// Private helpers for reading individual parts of the manifest.
namespace {

// An L1 region as capture reads it: its elements' bytes from its address.
MultiSampledRegion make_l1_read_region(const manifest::content::L1& l1) {
    return MultiSampledRegion{
        .kind = RegionKind::L1, .address = l1.address, .size_bytes = l1.num_elements * l1.element_size};
}

// A stream register as capture reads it: one word.
MultiSampledRegion make_register_read_region(const manifest::content::StreamReg& stream) {
    return MultiSampledRegion{.kind = RegionKind::REGISTER, .address = stream.address, .size_bytes = sizeof(uint32_t)};
}

// `values` ascending by `key`, each once.
template <typename T, typename Key = std::identity>
void sort_and_remove_duplicates(std::vector<T>& values, Key key = {}) {
    std::ranges::sort(values, {}, key);
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

PlannedRouter collect_planned_router(const manifest::Chip& chip, const manifest::Router& router) {
    PlannedRouter planned{
        .path = manifest::router_path({.node = chip.fabric_node_id, .key = router.identity.key}),
        .eth_chan = router.identity.eth_chan,
        .virtual_core =
            {
                .x = static_cast<uint32_t>(router.identity.virtual_core.x),
                .y = static_cast<uint32_t>(router.identity.virtual_core.y),
            },
    };
    // A region several ERISCs' fields name comes once per ERISC, so both lists are deduplicated below.
    manifest::for_each_router_region(
        router,
        ttsl::overloaded{
            [&](std::string_view /*path*/, const manifest::content::L1& l1) {
                if (l1.multi_sampled.has_value()) {
                    planned.multi_sampled_regions.push_back(make_l1_read_region(l1));
                }
            },
            [&](std::string_view /*path*/, const manifest::content::StreamReg& stream) {
                planned.stream_register_addresses.push_back(stream.address);
                if (stream.multi_sampled.has_value()) {
                    planned.multi_sampled_regions.push_back(make_register_read_region(stream));
                }
            },
        });
    sort_and_remove_duplicates(planned.stream_register_addresses);
    sort_and_remove_duplicates(planned.multi_sampled_regions, &MultiSampledRegion::address);
    return planned;
}

// The arch areas the manifest marks multi_sampled, in ArchArea's order.
std::vector<MultiSampledRegion> collect_multi_sampled_arch_areas(const manifest::Arch& arch) {
    std::vector<MultiSampledRegion> sampled;
    manifest::for_each_arch_area(arch, [&](std::string_view /*name*/, const manifest::content::L1& area) {
        if (area.multi_sampled.has_value()) {
            sampled.push_back(make_l1_read_region(area));
        }
    });
    return sampled;
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
                .path = manifest::chip_path(chip.fabric_node_id),
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
        .multi_sampled_arch_areas = collect_multi_sampled_arch_areas(facts.arch),
        .core_registers = collect_core_registers(facts.arch.core_registers),
        .chips = collect_planned_chips(facts),
    };
}

}  // namespace tt::tt_fabric::capture
