// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_routing.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_routing_table.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_paths.hpp"
#include "tt_metal/fabric/routing_2d_table_builder.hpp"

namespace tt::tt_fabric::decode {

namespace {

// The bytes that are the chip's: everything past state_manager.
std::span<const std::byte> get_common_routing_table_bytes(const routing_l1_info_t& table) {
    return std::as_bytes(std::span(&table, 1)).subspan(offsetof(routing_l1_info_t, my_mesh_id));
}

// The bytes every chip of a mesh should share: the 2D action vectors, or the 1D path table.
std::span<const std::byte> get_mesh_member_bytes(const routing_l1_info_t& table, bool is_2d) {
    if (is_2d) {
        return std::as_bytes(std::span(table.route_table_2d.action_vectors));
    }
    return std::as_bytes(std::span(&table.routing_path_table_1d, 1));
}

Finding make_finding(FindingCode code, std::vector<std::string> paths, std::string message) {
    return Finding{.code = code, .paths = std::move(paths), .message = std::move(message)};
}

// A chip's groups of routers, as "[E0, W0] [N0]", for the message.
std::string describe_copies(const ChipRouting& routing) {
    std::vector<std::string> groups;
    for (const RoutingTableCopy& copy : routing.copies) {
        groups.push_back(fmt::format("[{}]", fmt::join(copy.routers, ", ")));
    }
    return fmt::format("{}", fmt::join(groups, " "));
}

// A copy whose mesh id, device id or coordinates aren't the manifest chip's. The coordinates are compared in 2D only.
void add_identity_finding(
    const RoutingTableCopy& copy, const manifest::Chip& chip, bool is_2d, std::vector<Finding>& findings) {
    const routing_l1_info_t& table = copy.read.table;
    const FabricNodeId node = chip.fabric_node_id;
    const bool has_coord = is_2d && chip.mesh_coord.size() == 2;
    const bool coord_matches =
        !has_coord || (table.my_mesh_coord_y == chip.mesh_coord[0] && table.my_mesh_coord_x == chip.mesh_coord[1]);
    if (table.my_mesh_id == *node.mesh_id && table.my_device_id == node.chip_id && coord_matches) {
        return;
    }
    findings.push_back(make_finding(
        FindingCode::ROUTING_TABLE_IDENTITY_MISMATCH,
        copy.routers,
        fmt::format(
            "the table says mesh {} chip {} at ({}, {}); the manifest's chip is mesh {} chip {}{}",
            table.my_mesh_id,
            table.my_device_id,
            table.my_mesh_coord_y,
            table.my_mesh_coord_x,
            *node.mesh_id,
            node.chip_id,
            has_coord ? fmt::format(" at ({}, {})", chip.mesh_coord[0], chip.mesh_coord[1]) : std::string{})));
}

// A 2D copy whose shape can't be a 2D route table's (its vectors and trees are then null), or isn't its mesh's.
void add_shape_findings(const RoutingTableCopy& copy, const manifest::Mesh& mesh, std::vector<Finding>& findings) {
    const routing_l1_info_t& table = copy.read.table;
    if (!is_valid_2d_route_table_shape(table.mesh_y_size, table.mesh_x_size)) {
        findings.push_back(make_finding(
            FindingCode::ROUTING_TABLE_SHAPE_INVALID,
            copy.routers,
            fmt::format(
                "the table's mesh shape {}x{} can't be a 2D route table's", table.mesh_y_size, table.mesh_x_size)));
        return;
    }
    const bool is_manifest_shape =
        mesh.shape.size() == 2 && table.mesh_y_size == mesh.shape[0] && table.mesh_x_size == mesh.shape[1];
    if (!is_manifest_shape) {
        findings.push_back(make_finding(
            FindingCode::ROUTING_TABLE_SHAPE_MISMATCH,
            copy.routers,
            fmt::format(
                "the table's mesh shape is {}x{}; the manifest's is {}",
                table.mesh_y_size,
                table.mesh_x_size,
                fmt::join(mesh.shape, "x"))));
    }
}

void add_chip_findings(
    const ChipRouting& routing, const manifest::ManifestFacts& facts, std::vector<Finding>& findings) {
    if (routing.copies.size() > 1) {
        std::vector<std::string> routers;
        for (const RoutingTableCopy& copy : routing.copies) {
            routers.insert(routers.end(), copy.routers.begin(), copy.routers.end());
        }
        findings.push_back(make_finding(
            FindingCode::ROUTING_TABLE_COPIES_DIFFER,
            std::move(routers),
            fmt::format(
                "the routers of {} hold {} different tables: {}",
                manifest::chip_path(routing.chip),
                routing.copies.size(),
                describe_copies(routing))));
    }
    const bool is_2d = facts.fabric.is_2d_routing;
    const manifest::Chip* chip = manifest::find_chip(facts, routing.chip);
    const manifest::Mesh* mesh = manifest::find_mesh(facts, routing.chip.mesh_id);
    for (const RoutingTableCopy& copy : routing.copies) {
        if (chip != nullptr) {
            add_identity_finding(copy, *chip, is_2d, findings);
        }
        if (is_2d && mesh != nullptr) {
            add_shape_findings(copy, *mesh, findings);
        }
    }
}

// Each mesh's chips' first copies against the mesh's first chip's, over the members every chip of a mesh should share.
void add_mesh_member_findings(const RoutingAnalysis& analysis, bool is_2d, std::vector<Finding>& findings) {
    const std::vector<ChipRouting>& chips = analysis.chips;
    for (size_t first = 0; first < chips.size(); ++first) {
        const MeshId mesh_id = chips[first].chip.mesh_id;
        const bool is_mesh_first = std::none_of(chips.begin(), chips.begin() + first, [&](const ChipRouting& other) {
            return other.chip.mesh_id == mesh_id;
        });
        if (!is_mesh_first) {
            continue;
        }
        const std::span<const std::byte> expected =
            get_mesh_member_bytes(chips[first].copies.front().read.table, is_2d);
        std::vector<std::string> differing;
        for (size_t other = first + 1; other < chips.size(); ++other) {
            const bool is_same_mesh = chips[other].chip.mesh_id == mesh_id;
            if (is_same_mesh &&
                !std::ranges::equal(expected, get_mesh_member_bytes(chips[other].copies.front().read.table, is_2d))) {
                differing.push_back(manifest::chip_path(chips[other].chip));
            }
        }
        if (!differing.empty()) {
            findings.push_back(make_finding(
                FindingCode::ROUTING_TABLE_MESH_MEMBERS_DIFFER,
                differing,
                fmt::format(
                    "the {} of {} differs from {}'s",
                    is_2d ? "2D action vectors" : "1D path table",
                    fmt::join(differing, ", "),
                    manifest::chip_path(chips[first].chip))));
        }
    }
}

}  // namespace

std::optional<RoutingTableRead> find_routing_table_read(const RouterValues& values) {
    const RegionRead<routing_l1_info_t>* read =
        values.find_read<routing_l1_info_t>(make_arch_area_path("routing_table"), std::nullopt);
    if (read == nullptr || read->elements.empty()) {
        return std::nullopt;
    }
    return RoutingTableRead{.table = read->elements.front(), .at = read->bin_offset};
}

RoutingAnalysis analyze_routing_tables(
    std::span<const RouterRoutingTable> tables, const manifest::ManifestFacts& facts, std::vector<Finding>& findings) {
    RoutingAnalysis analysis;
    for (const RouterRoutingTable& entry : tables) {
        auto chip = std::ranges::find(analysis.chips, entry.ref.node, &ChipRouting::chip);
        if (chip == analysis.chips.end()) {
            analysis.chips.push_back(ChipRouting{.chip = entry.ref.node});
            chip = std::prev(analysis.chips.end());
        }
        const std::span<const std::byte> bytes = get_common_routing_table_bytes(entry.read.table);
        auto copy = std::ranges::find_if(chip->copies, [&](const RoutingTableCopy& other) {
            return std::ranges::equal(get_common_routing_table_bytes(other.read.table), bytes);
        });
        if (copy == chip->copies.end()) {
            chip->copies.push_back(RoutingTableCopy{
                .read = entry.read,
                .read_router_ref = entry.ref,
                .read_manifest_router = entry.router});
            copy = std::prev(chip->copies.end());
        }
        copy->routers.push_back(entry.path);
        analysis.copy_by_router[entry.path] = static_cast<uint32_t>(std::distance(chip->copies.begin(), copy));
    }
    for (const ChipRouting& chip : analysis.chips) {
        add_chip_findings(chip, facts, findings);
    }
    add_mesh_member_findings(analysis, facts.fabric.is_2d_routing, findings);
    return analysis;
}

json make_chip_routing_json(const ChipRouting& routing, const manifest::ManifestFacts& facts) {
    json copies = json::array();
    for (const RoutingTableCopy& copy : routing.copies) {
        const InterpretContext context =
            make_interpret_context(facts, copy.read_router_ref, *copy.read_manifest_router, copy.routers.front());
        json entry;
        entry["routers"] = copy.routers;
        entry["read_router"] = copy.routers.front();
        entry["at"] = copy.read.at;
        entry["table"] = make_chip_routing_table_json(copy.read.table, context);
        copies.push_back(std::move(entry));
    }
    json out;
    out["copies"] = std::move(copies);
    return out;
}

json make_router_routing_table_json(std::optional<uint32_t> copy) {
    if (!copy.has_value()) {
        return json(nullptr);
    }
    json out;
    out["copy"] = *copy;
    return out;
}

}  // namespace tt::tt_fabric::decode
