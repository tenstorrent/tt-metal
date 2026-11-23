// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_route_view.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include <fmt/format.h>
#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_fields.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_paths.hpp"

namespace tt::tt_fabric::decode {

namespace {

constexpr std::array<eth_chan_directions, 4> k_cardinal_directions{
    eth_chan_directions::EAST, eth_chan_directions::WEST, eth_chan_directions::NORTH, eth_chan_directions::SOUTH};

// The index of `coord` in RouteMesh::chip_by_coord.
size_t get_coord_index(const RouteMesh& mesh, MeshCoord coord) {
    return (size_t{coord.y} * mesh.num_columns) + coord.x;
}

// The cardinal directions that step from `coord` to another coordinate of `mesh`, one bit per direction.
uint8_t find_step_directions(const RouteMesh& mesh, MeshCoord coord) {
    uint8_t directions = 0;
    for (const eth_chan_directions direction : k_cardinal_directions) {
        if (step_mesh_coord(mesh, coord, direction).has_value()) {
            directions |= Routing2DCodec::action_bit(direction);
        }
    }
    return directions;
}

// The mesh's shape and torus, with every chip the manifest lists placed at its mesh_coord.
RouteMesh describe_route_mesh(const manifest::Mesh& mesh) {
    TT_FATAL(
        mesh.shape.size() == 2,
        "Fabric decode: mesh {} has a {}-dimensional shape; routes are walked on 2D meshes",
        *mesh.mesh_id,
        mesh.shape.size());
    RouteMesh out{
        .mesh_id = mesh.mesh_id,
        .num_rows = mesh.shape[0],
        .num_columns = mesh.shape[1],
        .torus = mesh.torus.value_or(manifest::MeshTorus{}),
        .express_axis = mesh.express_axis,
    };
    out.chip_by_coord.resize(size_t{out.num_rows} * out.num_columns);
    for (const manifest::Chip& chip : mesh.chips) {
        const std::string path = manifest::chip_path(chip.fabric_node_id);
        TT_FATAL(chip.mesh_coord.size() == 2, "Fabric decode: chip {}'s mesh_coord isn't 2D", path);
        const MeshCoord coord{.y = chip.mesh_coord[0], .x = chip.mesh_coord[1]};
        TT_FATAL(
            coord.y < out.num_rows && coord.x < out.num_columns,
            "Fabric decode: chip {}'s mesh_coord ({}, {}) is off its {}x{} mesh",
            path,
            coord.y,
            coord.x,
            out.num_rows,
            out.num_columns);
        std::optional<LogicalChipId>& placed_chip = out.chip_by_coord[get_coord_index(out, coord)];
        TT_FATAL(
            !placed_chip.has_value(),
            "Fabric decode: chips {} and {} share a mesh_coord",
            manifest::chip_path(FabricNodeId(mesh.mesh_id, placed_chip.value_or(0))),
            path);
        placed_chip = chip.fabric_node_id.chip_id;
    }
    // FabricContext counts the corner's neighbors over ControlPlane's links; here they're its coordinate steps, which
    // the mesh's shape and torus give. Z isn't counted: only 1D reads this, and 1D has no express links.
    out.wraps_around =
        out.num_rows > 0 && out.num_columns > 0 && std::popcount(find_step_directions(out, MeshCoord{})) == 2;
    return out;
}

// The value `read_erisc_value` reads for each ERISC that runs receiver `channel`, which they must agree on. Null when
// no ERISC runs it.
template <typename ReadEriscValue>
auto read_receiver_value(
    const manifest::ReceiverChannel& channel,
    std::string_view path,
    std::string_view what,
    const ReadEriscValue& read_erisc_value) -> std::optional<decltype(read_erisc_value(uint32_t{}))> {
    std::optional<decltype(read_erisc_value(uint32_t{}))> value;
    for (const uint32_t erisc : channel.serviced_by) {
        auto erisc_value = read_erisc_value(erisc);
        TT_FATAL(
            !value.has_value() || *value == erisc_value,
            "Fabric decode: receiver channel {}'s ERISCs give it different {}",
            path,
            what);
        value = std::move(erisc_value);
    }
    return value;
}

// The flag field `key` of ERISC `erisc` of a channel. False when the ERISC isn't fed it.
bool read_channel_flag(const manifest::FieldsByErisc& fields_by_erisc, uint32_t erisc, std::string_view key) {
    if (erisc >= fields_by_erisc.size()) {
        return false;
    }
    const auto field = std::ranges::find(fields_by_erisc[erisc], key, &manifest::Field::key);
    if (field == fields_by_erisc[erisc].end() || !field->content.has_value()) {
        return false;
    }
    const auto* flag = std::get_if<manifest::content::Flag>(&*field->content);
    return flag != nullptr && flag->value;
}

RouteReceiver describe_route_receiver(const manifest::ReceiverChannel& channel, std::string_view path) {
    const std::optional<std::optional<uint32_t>> forwards_on =
        read_receiver_value(channel, path, "forwarding VCs", [&](uint32_t erisc) {
            return erisc < channel.forwards_on_by_erisc.size() ? channel.forwards_on_by_erisc[erisc] : std::nullopt;
        });
    const std::optional<bool> intermesh_ingress =
        read_receiver_value(channel, path, "intermesh_ingress values", [&](uint32_t erisc) {
            return read_channel_flag(channel.fields_by_erisc, erisc, "intermesh_ingress");
        });
    return RouteReceiver{
        .forwards_on = forwards_on.value_or(std::nullopt), .intermesh_ingress = intermesh_ingress.value_or(false)};
}

RouteRouter describe_route_router(const manifest::RouterRef& ref, const manifest::Router& router) {
    const std::string path = manifest::router_path(ref);
    RouteRouter out{
        .ref = ref,
        .facing = find_router_facing(router),
        .peer = router.link.peer,
        .is_intermesh = router.link.edge_capability == EdgeCapability::INTERMESH,
    };
    for (uint32_t vc = 0; vc < router.channels.receivers.size(); ++vc) {
        std::vector<RouteReceiver>& receivers = out.receivers.emplace_back();
        for (uint32_t channel = 0; channel < router.channels.receivers[vc].size(); ++channel) {
            const std::string channel_path = fmt::format("{}/channels/receivers/vc{}/ch{}", path, vc, channel);
            receivers.push_back(describe_route_receiver(router.channels.receivers[vc][channel], channel_path));
        }
    }
    return out;
}

}  // namespace

eth_chan_directions find_router_facing(const manifest::Router& router) {
    const manifest::Field* field = manifest::find_erisc_field(router.fields_by_erisc, 0, "my_direction");
    const auto* direction = field != nullptr && field->content.has_value()
                                ? std::get_if<manifest::content::Enum>(&*field->content)
                                : nullptr;
    TT_FATAL(direction != nullptr, "Fabric decode: a router's manifest entry has no my_direction field");
    return static_cast<eth_chan_directions>(direction->value);
}

eth_chan_directions get_opposite_direction(eth_chan_directions direction) {
    switch (direction) {
        case eth_chan_directions::EAST: return eth_chan_directions::WEST;
        case eth_chan_directions::WEST: return eth_chan_directions::EAST;
        case eth_chan_directions::NORTH: return eth_chan_directions::SOUTH;
        case eth_chan_directions::SOUTH: return eth_chan_directions::NORTH;
        default: return direction;
    }
}

std::optional<MeshCoord> step_mesh_coord(const RouteMesh& mesh, MeshCoord coord, eth_chan_directions direction) {
    const bool along_y = direction == eth_chan_directions::NORTH || direction == eth_chan_directions::SOUTH;
    const bool along_x = direction == eth_chan_directions::EAST || direction == eth_chan_directions::WEST;
    if (!along_y && !along_x) {
        return std::nullopt;
    }
    const uint32_t axis_size = along_y ? mesh.num_rows : mesh.num_columns;
    const bool wraps = along_y ? mesh.torus.y : mesh.torus.x;
    const uint32_t position = along_y ? coord.y : coord.x;
    if (axis_size <= 1 || position >= axis_size) {
        return std::nullopt;
    }
    const bool increasing = direction == eth_chan_directions::SOUTH || direction == eth_chan_directions::EAST;
    uint32_t next = 0;
    if (increasing) {
        if (position + 1 < axis_size) {
            next = position + 1;
        } else if (wraps) {
            next = 0;
        } else {
            return std::nullopt;
        }
    } else {
        if (position > 0) {
            next = position - 1;
        } else if (wraps) {
            next = axis_size - 1;
        } else {
            return std::nullopt;
        }
    }
    return along_y ? MeshCoord{.y = next, .x = coord.x} : MeshCoord{.y = coord.y, .x = next};
}

RouteView::RouteView(const manifest::ManifestFacts& facts) {
    for (const manifest::Mesh& mesh : facts.meshes) {
        meshes_.push_back(describe_route_mesh(mesh));
        for (const manifest::Chip& chip : mesh.chips) {
            coord_by_chip_.emplace(chip.fabric_node_id, MeshCoord{.y = chip.mesh_coord[0], .x = chip.mesh_coord[1]});
            for (const manifest::Router& router : chip.routers) {
                const manifest::RouterRef ref{.node = chip.fabric_node_id, .key = router.identity.key};
                const size_t index = routers_.size();
                const RouteRouter& route_router = routers_.emplace_back(describe_route_router(ref, router));
                router_by_key_.emplace(std::tuple{ref.node, ref.key.direction, ref.key.routing_plane}, index);
                const bool facing_is_new =
                    router_by_facing_.emplace(std::tuple{ref.node, ref.key.routing_plane, route_router.facing}, index)
                        .second;
                TT_FATAL(
                    facing_is_new,
                    "Fabric decode: two routers of chip {} on routing plane {} face the same way, as {} does",
                    manifest::chip_path(ref.node),
                    ref.key.routing_plane,
                    manifest::router_path(ref));
            }
        }
    }
}

const RouteMesh* RouteView::find_mesh(MeshId mesh_id) const {
    const auto mesh = std::ranges::find(meshes_, mesh_id, &RouteMesh::mesh_id);
    return mesh == meshes_.end() ? nullptr : &*mesh;
}

std::optional<MeshCoord> RouteView::find_chip_coord(FabricNodeId chip) const {
    const auto coord = coord_by_chip_.find(chip);
    return coord == coord_by_chip_.end() ? std::nullopt : std::optional(coord->second);
}

std::optional<FabricNodeId> RouteView::find_chip_at(MeshId mesh_id, MeshCoord coord) const {
    const RouteMesh* mesh = find_mesh(mesh_id);
    if (mesh == nullptr || coord.y >= mesh->num_rows || coord.x >= mesh->num_columns) {
        return std::nullopt;
    }
    const std::optional<LogicalChipId>& chip_id = mesh->chip_by_coord[get_coord_index(*mesh, coord)];
    return chip_id.has_value() ? std::optional(FabricNodeId(mesh_id, *chip_id)) : std::nullopt;
}

std::optional<FabricNodeId> RouteView::find_next_chip(FabricNodeId chip, eth_chan_directions direction) const {
    const RouteMesh* mesh = find_mesh(chip.mesh_id);
    const std::optional<MeshCoord> coord = find_chip_coord(chip);
    if (mesh == nullptr || !coord.has_value()) {
        return std::nullopt;
    }
    const std::optional<MeshCoord> next = step_mesh_coord(*mesh, *coord, direction);
    return next.has_value() ? find_chip_at(chip.mesh_id, *next) : std::nullopt;
}

uint8_t RouteView::find_neighbor_directions(FabricNodeId chip) const {
    uint8_t directions = 0;
    for (const eth_chan_directions direction : k_cardinal_directions) {
        if (find_next_chip(chip, direction).has_value()) {
            directions |= Routing2DCodec::action_bit(direction);
        }
    }
    return directions;
}

const RouteRouter* RouteView::find_router(const manifest::RouterRef& ref) const {
    const auto index = router_by_key_.find(std::tuple{ref.node, ref.key.direction, ref.key.routing_plane});
    return index == router_by_key_.end() ? nullptr : &routers_[index->second];
}

const RouteRouter* RouteView::find_receiving_router(
    FabricNodeId chip, routing_plane_id_t routing_plane, eth_chan_directions facing) const {
    const auto index = router_by_facing_.find(std::tuple{chip, routing_plane, facing});
    return index == router_by_facing_.end() ? nullptr : &routers_[index->second];
}

}  // namespace tt::tt_fabric::decode
