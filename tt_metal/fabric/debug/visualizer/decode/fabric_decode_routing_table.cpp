// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_routing_table.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <numeric>
#include <optional>
#include <string>

#include <enchantum/enchantum.hpp>

#include "tt_metal/fabric/routing_2d_table_builder.hpp"

namespace tt::tt_fabric::decode {

namespace detail {

json make_direction_table_entry_json(uint8_t original_direction) {
    std::optional<std::string> name;
    if (original_direction < eth_chan_directions::COUNT) {
        name = find_enum_name(static_cast<eth_chan_directions>(original_direction));
    } else {
        name = find_enum_name(static_cast<eth_chan_magic_values>(original_direction));
    }
    return name.has_value() ? json(*name) : json(nullptr);
}

json make_1d_hop_action_json(RoutingFieldsConstants::LowLatency::HopAction action) {
    return json(find_enum_name(action).value_or(std::string{}));
}

json make_2d_axis_code_json(Fabric2DAxis axis, uint8_t code) {
    const uint8_t action = Routing2DCodec::widen(axis, code);
    if (action == 0) {
        return json("stop");
    }
    const auto direction = static_cast<eth_chan_directions>(std::countr_zero(action));
    return json(find_enum_name(direction).value_or(std::string{}));
}

}  // namespace detail

json make_direction_table_entry_vocabulary_json() {
    json names = json::array();
    for (uint8_t direction = 0; direction < eth_chan_directions::COUNT; ++direction) {
        names.push_back(detail::make_direction_table_entry_json(direction));
    }
    names.push_back(detail::make_direction_table_entry_json(eth_chan_magic_values::INVALID_DIRECTION));
    names.push_back(detail::make_direction_table_entry_json(eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY));
    return names;
}

json make_2d_axis_code_vocabulary_json() {
    json names = json::array({"stop"});
    for (uint8_t direction = 0; direction < eth_chan_directions::COUNT; ++direction) {
        names.push_back(detail::make_direction_table_entry_json(direction));
    }
    return names;
}

json make_1d_hop_action_vocabulary_json() {
    json names = json::array();
    for (const auto action : enchantum::values<RoutingFieldsConstants::LowLatency::HopAction>) {
        names.push_back(detail::make_1d_hop_action_json(action));
    }
    return names;
}

namespace {

json make_chip_identity_json(const routing_l1_info_t& table) {
    json out;
    out["mesh_id"] = table.my_mesh_id;
    out["device_id"] = table.my_device_id;
    out["mesh_coord"] = json::array({table.my_mesh_coord_y, table.my_mesh_coord_x});
    out["mesh_shape"] = json::array({table.mesh_y_size, table.mesh_x_size});
    return out;
}

// The first hop to each chip of the router's mesh, by chip id. Past the mesh's chips the table's entries are unwritten.
json make_intra_mesh_direction_table_json(const routing_l1_info_t& table, const manifest::Mesh& mesh) {
    const uint64_t num_chips = std::accumulate(mesh.shape.begin(), mesh.shape.end(), uint64_t{1}, std::multiplies<>{});
    const auto num_entries = static_cast<uint32_t>(std::min<uint64_t>(num_chips, MAX_MESH_SIZE));
    json out = json::array();
    for (uint32_t chip = 0; chip < num_entries; ++chip) {
        const uint8_t direction = table.intra_mesh_direction_table.get_original_direction(chip);
        out.push_back(detail::make_direction_table_entry_json(direction));
    }
    return out;
}

// The first hop to each mesh with a route entry: unwritten entries are left out.
json make_inter_mesh_direction_table_json(const routing_l1_info_t& table) {
    json out = json::object();
    for (uint32_t mesh = 0; mesh < MAX_NUM_MESHES; ++mesh) {
        const uint8_t direction = table.inter_mesh_direction_table.get_original_direction(mesh);
        if (direction != eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY) {
            out[std::to_string(mesh)] = detail::make_direction_table_entry_json(direction);
        }
    }
    return out;
}

// The chip to leave this mesh from, to each mesh with one.
json make_2d_exit_node_table_json(const routing_l1_info_t& table) {
    json out = json::object();
    for (uint32_t mesh = 0; mesh < MAX_NUM_MESHES; ++mesh) {
        if (table.exit_node_table[mesh] != eth_chan_magic_values::INVALID_ROUTING_TABLE_ENTRY) {
            out[std::to_string(mesh)] = table.exit_node_table[mesh];
        }
    }
    return out;
}

// `edge_count` edges of one axis's tree, from `region`, in stored order.
json make_2d_mcast_tree_json(const uint8_t* region, uint32_t edge_count, Fabric2DAxis axis) {
    json out = json::array();
    for (uint32_t edge = 0; edge < edge_count; ++edge) {
        const uint16_t packed = Routing2DCodec::get_mcast_tree_edge(region, edge);
        out.push_back(
            {{"child", Routing2DCodec::mcast_edge_child(packed)},
             {"parent", Routing2DCodec::mcast_edge_parent(packed)},
             {"direction", detail::make_2d_axis_code_json(axis, Routing2DCodec::mcast_edge_output(packed))}});
    }
    return out;
}

// This chip's Y tree, then its X tree, which follows the Y tree's edges. Null when the table's shape can't be a 2D one.
json make_2d_mcast_trees_json(const routing_l1_info_t& table) {
    const uint32_t y_size = table.mesh_y_size;
    const uint32_t x_size = table.mesh_x_size;
    if (!is_valid_2d_route_table_shape(y_size, x_size)) {
        return json(nullptr);
    }
    const uint8_t* y_tree = table.route_table_2d.mcast_trees;
    const uint8_t* x_tree = y_tree + Routing2DCodec::mcast_tree_x_offset(y_size);
    json out;
    out["y"] = make_2d_mcast_tree_json(y_tree, Routing2DCodec::mcast_tree_edge_count(y_size), Fabric2DAxis::Y);
    out["x"] = make_2d_mcast_tree_json(x_tree, Routing2DCodec::mcast_tree_edge_count(x_size), Fabric2DAxis::X);
    return out;
}

// The action vectors: `y` and `x`, each a row per destination coordinate, each row a step per current coordinate. Null
// when the table's shape can't be a 2D one.
json make_2d_action_vectors_json(const routing_l1_info_t& table) {
    const uint32_t y_size = table.mesh_y_size;
    const uint32_t x_size = table.mesh_x_size;
    if (!is_valid_2d_route_table_shape(y_size, x_size)) {
        return json(nullptr);
    }
    const uint8_t* vectors = table.route_table_2d.action_vectors;
    json y_rows = json::array();
    for (uint32_t dst_y = 0; dst_y < y_size; ++dst_y) {
        const uint8_t* row = Routing2DCodec::y_row(vectors, y_size, dst_y);
        json steps = json::array();
        for (uint32_t cur_y = 0; cur_y < y_size; ++cur_y) {
            const uint8_t code = Routing2DCodec::get_action_2bit(row, cur_y);
            steps.push_back(detail::make_2d_axis_code_json(Fabric2DAxis::Y, code));
        }
        y_rows.push_back(std::move(steps));
    }
    json x_rows = json::array();
    for (uint32_t dst_x = 0; dst_x < x_size; ++dst_x) {
        const uint8_t* row = Routing2DCodec::x_row(vectors, y_size, x_size, dst_x);
        json steps = json::array();
        for (uint32_t cur_x = 0; cur_x < x_size; ++cur_x) {
            const uint8_t code = Routing2DCodec::get_action_2bit(row, cur_x);
            steps.push_back(detail::make_2d_axis_code_json(Fabric2DAxis::X, code));
        }
        x_rows.push_back(std::move(steps));
    }
    json out;
    out["y"] = std::move(y_rows);
    out["x"] = std::move(x_rows);
    return out;
}

// One 1D entry's hop actions, hop 0 first, up to the first NOOP. The entry is `num_words` little-endian words, each
// holding BASE_HOPS hops of FIELD_WIDTH bits, LSB first, as the router consumes them.
json make_1d_path_entry_actions_json(const uint8_t* entry, uint32_t num_words) {
    using LowLatency = RoutingFieldsConstants::LowLatency;
    json actions = json::array();
    for (uint32_t hop = 0; hop < num_words * LowLatency::BASE_HOPS; ++hop) {
        const uint32_t bit = hop * LowLatency::FIELD_WIDTH;
        const uint32_t action = (entry[bit / 8] >> (bit % 8)) & LowLatency::FIELD_MASK;
        if (action == LowLatency::NOOP) {
            break;
        }
        actions.push_back(detail::make_1d_hop_action_json(static_cast<LowLatency::HopAction>(action)));
    }
    return actions;
}

// The 1D path table: each nonzero entry's index (its position in the table, which is the distance to its destination),
// hop count (how many actions it holds, at most what the entry's words fit) and actions, and how many entries are zero.
// Null when the manifest has no 1D stride.
json make_1d_path_table_json(const routing_l1_info_t& table, const InterpretContext& context) {
    if (!context.fabric.routing_1d_extension_words.has_value()) {
        return json(nullptr);
    }
    const uint32_t num_words = 1 + *context.fabric.routing_1d_extension_words;
    const uint32_t stride_bytes = num_words * sizeof(uint32_t);
    const uint32_t num_entries = std::min<uint32_t>(
        MAX_CHIPS_LOWLAT_1D, sizeof(table.routing_path_table_1d.paths) / std::max<uint32_t>(stride_bytes, 1));
    json entries = json::array();
    uint32_t num_zero_entries = 0;
    for (uint32_t index = 0; index < num_entries; ++index) {
        const uint8_t* entry = table.routing_path_table_1d.paths + index * stride_bytes;
        if (std::all_of(entry, entry + stride_bytes, [](uint8_t byte) { return byte == 0; })) {
            ++num_zero_entries;
            continue;
        }
        json actions = make_1d_path_entry_actions_json(entry, num_words);
        entries.push_back({{"index", index}, {"hops", actions.size()}, {"actions", std::move(actions)}});
    }
    json out;
    out["entries"] = std::move(entries);
    out["num_zero_entries"] = num_zero_entries;
    return out;
}

}  // namespace

json make_chip_routing_table_json(const routing_l1_info_t& table, const InterpretContext& context) {
    json out;
    out["identity"] = make_chip_identity_json(table);
    out["intra_mesh_directions"] = make_intra_mesh_direction_table_json(table, context.mesh);
    out["inter_mesh_directions"] = make_inter_mesh_direction_table_json(table);
    const bool is_2d = context.fabric.is_2d_routing;
    out["exit_nodes"] = is_2d ? make_2d_exit_node_table_json(table) : json(nullptr);
    out["action_vectors"] = is_2d ? make_2d_action_vectors_json(table) : json(nullptr);
    out["multicast_trees"] = is_2d ? make_2d_mcast_trees_json(table) : json(nullptr);
    out["path_table_1d"] = is_2d ? json(nullptr) : make_1d_path_table_json(table, context);
    return out;
}

}  // namespace tt::tt_fabric::decode
