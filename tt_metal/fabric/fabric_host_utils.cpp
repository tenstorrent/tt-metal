// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "fabric_host_utils.hpp"

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>
#include <tt_stl/assert.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include "erisc_datamover_builder.hpp"
#include <set>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>
#include "fabric_context.hpp"
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <fstream>
#include <type_traits>
#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <tt-logger/tt-logger.hpp>
#include <llrt/tt_cluster.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {

namespace {

// Mock cluster mapping export uses cluster descriptor filenames (basename). Strip MPI-rank uniquifier
// suffix appended during PSD discovery when multiple ranks share the same descriptor basename.
HostName hostname_for_mapping_export(const HostName& hostname) {
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_mock_enabled()) {
        return hostname;
    }
    constexpr std::string_view cluster_desc_suffix = ".yaml";
    const auto pos = hostname.rfind(cluster_desc_suffix);
    if (pos == std::string::npos || pos + cluster_desc_suffix.size() >= hostname.size()) {
        return hostname;
    }
    const std::string tail = hostname.substr(pos + cluster_desc_suffix.size());
    if (tail.size() <= 1 || tail.front() != '_') {
        return hostname;
    }
    for (char c : tail.substr(1)) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return hostname;
        }
    }
    return hostname.substr(0, pos + cluster_desc_suffix.size());
}

// Retruns a string representation of the enum value. 
// 
// enchantum::to_string yields an empty view for a value that is not a named enumerator, which happens for
// bitmask combinations of FabricType such as MESH|TORUS_X, so in those cases we fall back to the numeric value.
template <typename E>
std::string enum_name(E value) {
    const auto name = enchantum::to_string(value);
    if (name.empty()) {
        return std::to_string(static_cast<std::underlying_type_t<E>>(value));
    }
    return std::string(name);
}

// (mesh, chip, channel) is the stable identity for a fabric router. The debug snapshot artifact keys its
// live values on the same triple, so the viewer can join a snapshot onto a manifest.
nlohmann::ordered_json fabric_debug_endpoint_json(const FabricNodeId& node, chan_id_t chan) {
    nlohmann::ordered_json endpoint;
    endpoint["mesh_id"] = *node.mesh_id;
    endpoint["chip_id"] = node.chip_id;
    endpoint["eth_chan"] = chan;
    return endpoint;
}

}  // namespace

bool is_tt_fabric_config(tt::tt_fabric::FabricConfig fabric_config) {
    return is_1d_fabric_config(fabric_config) || is_2d_fabric_config(fabric_config);
}

FabricType get_fabric_type(tt::tt_fabric::FabricConfig fabric_config, bool is_ubb_galaxy) {
    switch (fabric_config) {
        // Issue: 32146, Special case for T3k WH devices to use Mesh fabric type instead of Torus_XY
        // WH T3K currently do not support Torus_XY fabric type, because they do not have wrapping connections.
        // If you want to use 1D Ring on t3k please use 1x8 MGD.
        case tt::tt_fabric::FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE:
        case tt::tt_fabric::FabricConfig::FABRIC_1D_RING: {
            if (is_ubb_galaxy) {
                return FabricType::TORUS_XY;
            }
            return FabricType::MESH;
        }
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X: return FabricType::TORUS_X;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y: return FabricType::TORUS_Y;
        case tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY: return FabricType::TORUS_XY;
        default: return FabricType::MESH;
    }
}

bool requires_more_connectivity(FabricType requested_type, FabricType available_type, const MeshShape& mesh_shape) {
    for (uint32_t axis = 0; axis < 2; ++axis) {
        if (has_genuine_torus_axis(requested_type, mesh_shape, axis) &&
            !has_genuine_torus_axis(available_type, mesh_shape, axis)) {
            return true;
        }
    }
    return false;
}

uint32_t compute_max_1d_hops(const std::vector<MeshShape>& mesh_shapes) {
    if (mesh_shapes.empty()) {
        return 0;
    }

    uint32_t max_dimension = 0;
    for (const auto& shape : mesh_shapes) {
        // For 1D routing, find the maximum dimension (either rows or cols)
        // Hops = max_dimension - 1 (e.g., 8 chips in a line = 7 hops)
        uint32_t rows = shape[0];
        uint32_t cols = shape[1];
        uint32_t mesh_max_dim = std::max(rows, cols);
        max_dimension = std::max(max_dimension, mesh_max_dim);
    }

    return (max_dimension > 0) ? (max_dimension - 1) : 0;
}

uint32_t compute_max_2d_hops(const std::vector<MeshShape>& mesh_shapes) {
    if (mesh_shapes.empty()) {
        return 0;
    }

    uint32_t max_hops = 0;
    for (const auto& shape : mesh_shapes) {
        // For 2D routing, compute Manhattan distance from corner to corner
        // Hops = (rows - 1) + (cols - 1)
        uint32_t rows = shape[0];
        uint32_t cols = shape[1];
        uint32_t mesh_hops = (rows - 1) + (cols - 1);
        max_hops = std::max(max_hops, mesh_hops);
    }

    return max_hops;
}

std::vector<uint32_t> get_forwarding_link_indices_in_direction(
    const ControlPlane& control_plane,
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    RoutingDirection direction) {
    const std::vector<chan_id_t>& fabric_channels =
        control_plane.get_active_fabric_eth_channels_in_direction(src_fabric_node_id, direction);

    // the subset of routers that support forwarding b/w those chips
    std::vector<chan_id_t> forwarding_channels;
    forwarding_channels =
        control_plane.get_forwarding_eth_chans_to_chip(src_fabric_node_id, dst_fabric_node_id, direction);

    std::vector<uint32_t> link_indices;
    link_indices.reserve(forwarding_channels.size());
    for (uint32_t i = 0; i < fabric_channels.size(); i++) {
        if (std::find(forwarding_channels.begin(), forwarding_channels.end(), fabric_channels[i]) !=
            forwarding_channels.end()) {
            link_indices.push_back(i);
        }
    }

    return link_indices;
}

void serialize_mesh_coordinates_to_file(
    const TopologyMapper& topology_mapper, const std::filesystem::path& output_file_path) {
    // Ensure output directory exists
    std::filesystem::create_directories(output_file_path.parent_path());

    // Get the mapping from TopologyMapper
    const auto& mapping = topology_mapper.get_local_logical_mesh_chip_id_to_physical_chip_id_mapping();
    const auto& mesh_graph = topology_mapper.get_mesh_graph();

    // Write to file using emitter with Flow style for inline sequences
    std::ofstream out_file(output_file_path);
    if (!out_file.is_open()) {
        TT_THROW("Failed to open output file: {}", output_file_path.string());
    }

    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "chips";
    emitter << YAML::Value << YAML::BeginMap;

    // Emit each chip with flow style for the coordinate array
    for (const auto& [fabric_node_id, physical_chip_id] : mapping) {
        MeshCoordinate mesh_coord = mesh_graph.chip_to_coordinate(fabric_node_id.mesh_id, fabric_node_id.chip_id);
        emitter << YAML::Key << physical_chip_id;
        emitter << YAML::Value;
        emitter << YAML::Flow << YAML::BeginSeq;
        for (size_t dim = 0; dim < mesh_coord.dims(); ++dim) {
            emitter << mesh_coord[dim];
        }
        emitter << YAML::EndSeq;
    }

    emitter << YAML::EndMap;
    emitter << YAML::EndMap;
    out_file << emitter.c_str();
    out_file.close();

    log_debug(tt::LogFabric, "Serialized physical chip mesh coordinate mapping to file: {}", output_file_path.string());
}

void serialize_asic_to_fabric_node_mapping_to_file(
    const TopologyMapper& topology_mapper, const std::filesystem::path& output_file_path) {
    // Ensure output directory exists
    std::filesystem::create_directories(output_file_path.parent_path());

    const auto& mesh_graph = topology_mapper.get_mesh_graph();
    const auto& physical_system_descriptor = topology_mapper.get_physical_system_descriptor();

    // Structure: hostname -> mesh_id -> umd_chip_id -> {asic_position, fabric_node_id, asic_id}
    struct AsicMapping {
        tt::tt_metal::TrayID tray_id;
        tt::tt_metal::ASICLocation asic_location;
        FabricNodeId fabric_node_id;
        tt::tt_metal::AsicID asic_id;
    };
    std::map<HostName, std::map<MeshId, std::map<ChipId, AsicMapping>>> mappings_by_host_mesh_and_chip;

    // Iterate through all meshes
    for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
        // Iterate through all fabric nodes in this mesh
        for (const auto& [_, chip_id] : mesh_graph.get_chip_ids(mesh_id)) {
            FabricNodeId fabric_node_id(mesh_id, chip_id);

            try {
                // Get ASIC ID for this fabric node
                tt::tt_metal::AsicID asic_id = topology_mapper.get_asic_id_from_fabric_node_id(fabric_node_id);

                // Get physical chip ID (UMD chip ID) for this fabric node
                ChipId umd_chip_id = topology_mapper.get_physical_chip_id_from_fabric_node_id(fabric_node_id);

                // Get ASIC position (tray_id and asic_location) from physical system descriptor
                tt::tt_metal::TrayID tray_id = physical_system_descriptor.get_tray_id(asic_id);
                tt::tt_metal::ASICLocation asic_location = physical_system_descriptor.get_asic_location(asic_id);

                // Get hostname for this fabric node (mock: cluster descriptor filename)
                HostName hostname =
                    hostname_for_mapping_export(topology_mapper.get_hostname_for_fabric_node_id(fabric_node_id));

                // Add to the mapping structure, indexed by umd_chip_id (physical chip ID)
                AsicMapping mapping{tray_id, asic_location, fabric_node_id, asic_id};
                mappings_by_host_mesh_and_chip[hostname][mesh_id].emplace(umd_chip_id, mapping);
            } catch (...) {
                // Skip unmapped fabric nodes
                continue;
            }
        }
    }

    // Write to file using YAML emitter
    std::ofstream out_file(output_file_path);
    if (!out_file.is_open()) {
        TT_THROW("Failed to open output file: {}", output_file_path.string());
    }

    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "asic_to_fabric_node_mapping";
    emitter << YAML::Value;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "hostnames";
    emitter << YAML::Value << YAML::BeginSeq;

    // Emit each hostname as a list item
    for (const auto& [hostname, mesh_mappings] : mappings_by_host_mesh_and_chip) {
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "hostname";
        emitter << YAML::Value << hostname;

        // Emit mesh as a key with a list value
        emitter << YAML::Key << "mesh";
        emitter << YAML::Value << YAML::BeginSeq;

        // Emit each mesh within this hostname
        for (const auto& [mesh_id, chip_mappings] : mesh_mappings) {
            // First emit mesh entry
            emitter << YAML::BeginMap;
            emitter << YAML::Key << "mesh";
            emitter << YAML::Value << *mesh_id;
            emitter << YAML::EndMap;

            // Then emit chips entry
            emitter << YAML::BeginMap;
            emitter << YAML::Key << "chips";
            emitter << YAML::Value << YAML::BeginSeq;

            // Emit each umd_chip_id mapping (physical chip ID)
            for (const auto& [umd_chip_id, mapping] : chip_mappings) {
                emitter << YAML::BeginMap;

                // Emit umd_chip_id field
                emitter << YAML::Key << "umd_chip_id";
                emitter << YAML::Value << umd_chip_id;

                // Emit asic_position
                emitter << YAML::Key << "asic_position";
                emitter << YAML::Value;
                emitter << YAML::BeginMap;
                emitter << YAML::Key << "tray_id";
                emitter << YAML::Value << *mapping.tray_id;
                emitter << YAML::Key << "asic_location";
                emitter << YAML::Value << *mapping.asic_location;
                emitter << YAML::EndMap;

                // Emit fabric_node_id
                emitter << YAML::Key << "fabric_node_id";
                emitter << YAML::Value;
                emitter << YAML::BeginMap;
                emitter << YAML::Key << "mesh_id";
                emitter << YAML::Value << *mapping.fabric_node_id.mesh_id;
                emitter << YAML::Key << "chip_id";
                emitter << YAML::Value << mapping.fabric_node_id.chip_id;
                emitter << YAML::EndMap;

                // Emit asic_id as the last field
                emitter << YAML::Key << "asic_id";
                emitter << YAML::Value << *mapping.asic_id;

                emitter << YAML::EndMap;
            }

            emitter << YAML::EndSeq;
            emitter << YAML::EndMap;
        }

        emitter << YAML::EndSeq;
        emitter << YAML::EndMap;
    }

    emitter << YAML::EndSeq;
    emitter << YAML::EndMap;
    emitter << YAML::EndMap;
    out_file << emitter.c_str();
    out_file.close();

    log_debug(tt::LogFabric, "Serialized ASIC to Fabric node ID mapping to file: {}", output_file_path.string());
}

namespace {

std::optional<PhysicalGroupingDescriptor> load_pgd_if_regular_file(const std::filesystem::path& path) {
    if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path)) {
        log_info(tt::LogFabric, "Loaded physical groupings from: {}", path.string());
        return PhysicalGroupingDescriptor(path);
    }
    return std::nullopt;
}

std::vector<std::filesystem::path> build_physical_grouping_descriptor_search_paths(
    const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor) {
    const char* cluster_name_env = std::getenv("TT_CLUSTER_NAME");
    const std::string cluster_name = cluster_name_env != nullptr ? cluster_name_env : "";
    const char* tt_metal_home_env = std::getenv("TT_METAL_HOME");
    const std::string tt_metal_home = tt_metal_home_env != nullptr ? tt_metal_home_env : ".";

    std::vector<std::filesystem::path> search_paths;
    search_paths.reserve(3);
    if (!cluster_name.empty()) {
        search_paths.push_back(
            std::filesystem::path("/data/scaleout_configs") / cluster_name /
            (cluster_name + "_physical_grouping_descriptor.textproto"));
        search_paths.push_back(
            std::filesystem::path(tt_metal_home) / "tests" / "tt_metal" / "tt_fabric" / "physical_groupings" /
            (cluster_name + "_physical_grouping_descriptor.textproto"));
    }

    std::string arch_cluster_filename = "default_physical_grouping_descriptor.textproto";
    auto& context = tt::tt_metal::MetalContext::instance();
    const auto& cluster = context.get_cluster();
    const tt::tt_metal::ClusterType cluster_type = cluster.get_cluster_type();
    const tt::ARCH arch = cluster.arch();
    if (cluster_type == tt::tt_metal::ClusterType::GALAXY && arch == tt::ARCH::WORMHOLE_B0) {
        arch_cluster_filename = "wh_bh_rev_c_galaxy_physical_grouping_descriptor.textproto";
    } else if (
        (cluster_type == tt::tt_metal::ClusterType::BLACKHOLE_GALAXY || cluster.is_ubb_galaxy()) &&
        arch == tt::ARCH::BLACKHOLE) {
        if (physical_system_descriptor != nullptr && physical_system_descriptor->is_bh_galaxy_rev_c()) {
            arch_cluster_filename = "wh_bh_rev_c_galaxy_physical_grouping_descriptor.textproto";
        } else {
            arch_cluster_filename = "bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
        }
    } else if (cluster_type == tt::tt_metal::ClusterType::T3K && arch == tt::ARCH::WORMHOLE_B0) {
        arch_cluster_filename = "wh_t3k_physical_grouping_descriptor.textproto";
    }

    search_paths.push_back(
        std::filesystem::path(tt_metal_home) / "tests" / "tt_metal" / "tt_fabric" / "physical_groupings" /
        arch_cluster_filename);
    return search_paths;
}

}  // namespace

PhysicalGroupingDescriptor find_and_load_physical_grouping_descriptor(
    const std::optional<std::filesystem::path>& pgd_path,
    const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor) {
    if (pgd_path.has_value() && !pgd_path->empty()) {
        if (auto loaded = load_pgd_if_regular_file(*pgd_path)) {
            return *loaded;
        }
        TT_THROW("Physical Grouping Descriptor path provided but file does not exist: {}", pgd_path->string());
    }

    const char* pgd_path_env = std::getenv("TT_METAL_PHYSICAL_GROUPING_DESCRIPTOR_PATH");
    if (pgd_path_env != nullptr && std::strlen(pgd_path_env) > 0) {
        const std::filesystem::path explicit_path(pgd_path_env);
        if (auto loaded = load_pgd_if_regular_file(explicit_path)) {
            return *loaded;
        }
        TT_THROW(
            "TT_METAL_PHYSICAL_GROUPING_DESCRIPTOR_PATH is set but file does not exist: {}", explicit_path.string());
    }

    const auto search_paths = build_physical_grouping_descriptor_search_paths(physical_system_descriptor);
    for (const auto& path : search_paths) {
        if (auto loaded = load_pgd_if_regular_file(path)) {
            return *loaded;
        }
    }

    const char* cluster_name_env = std::getenv("TT_CLUSTER_NAME");
    std::string error_msg = "Could not find Physical Grouping Descriptor file. Searched:\n";
    for (const auto& path : search_paths) {
        error_msg += "  - " + path.string() + "\n";
    }
    if (cluster_name_env != nullptr && cluster_name_env[0] != '\0') {
        error_msg += std::string("Cluster name from TT_CLUSTER_NAME: ") + cluster_name_env + "\n";
    } else {
        error_msg += "TT_CLUSTER_NAME not set\n";
    }
    throw std::runtime_error(error_msg);
}

std::optional<PhysicalGroupingDescriptor> try_find_and_load_physical_grouping_descriptor(
    const std::optional<std::filesystem::path>& pgd_path,
    const tt::tt_metal::PhysicalSystemDescriptor* physical_system_descriptor) {
    try {
        return find_and_load_physical_grouping_descriptor(pgd_path, physical_system_descriptor);
    } catch (const std::exception& e) {
        log_debug(tt::LogFabric, "Physical Grouping Descriptor not loaded (soft-skip): {}", e.what());
        return std::nullopt;
    }
}

void serialize_intermesh_port_assignment_to_file(
    const std::map<FabricNodeId, std::unordered_map<chan_id_t, RoutingDirection>>& exit_node_directions,
    const std::map<FabricNodeId, std::unordered_map<chan_id_t, std::pair<FabricNodeId, chan_id_t>>>&
        intermesh_chan_to_peer,
    const std::filesystem::path& output_file_path) {
    auto dir_to_str = [](RoutingDirection d) -> const char* {
        switch (d) {
            case RoutingDirection::N: return "N";
            case RoutingDirection::E: return "E";
            case RoutingDirection::S: return "S";
            case RoutingDirection::W: return "W";
            case RoutingDirection::Z: return "Z";
            case RoutingDirection::C: return "C";
            default: return "NONE";
        }
    };

    std::map<std::string, std::vector<std::string>> intermesh_port_assignment;
    for (const auto& [my_fn, chan_map] : intermesh_chan_to_peer) {
        std::vector<chan_id_t> chans;
        chans.reserve(chan_map.size());
        for (const auto& [c, _peer] : chan_map) {
            chans.push_back(c);
        }
        std::sort(chans.begin(), chans.end());
        for (auto c : chans) {
            const auto& [peer_fn, peer_chan] = chan_map.at(c);
            RoutingDirection dir = RoutingDirection::NONE;
            if (auto dit = exit_node_directions.find(my_fn); dit != exit_node_directions.end()) {
                if (auto cit = dit->second.find(c); cit != dit->second.end()) {
                    dir = cit->second;
                }
            }
            intermesh_port_assignment[fmt::format("M{}->M{}", *my_fn.mesh_id, *peer_fn.mesh_id)].push_back(fmt::format(
                "D{}ch{}({})>M{}D{}ch{}",
                my_fn.chip_id,
                c,
                dir_to_str(dir),
                *peer_fn.mesh_id,
                peer_fn.chip_id,
                peer_chan));
        }
    }
    for (auto& [_boundary, entries] : intermesh_port_assignment) {
        std::sort(entries.begin(), entries.end());
    }

    std::filesystem::create_directories(output_file_path.parent_path());

    std::ofstream out_file(output_file_path);
    if (!out_file.is_open()) {
        TT_THROW("Failed to open output file: {}", output_file_path.string());
    }
    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "intermesh_port_assignment" << YAML::Value << YAML::BeginMap;
    for (const auto& [boundary, entries] : intermesh_port_assignment) {
        emitter << YAML::Key << boundary << YAML::Value << YAML::Flow << YAML::BeginSeq;
        for (const auto& entry : entries) {
            emitter << entry;
        }
        emitter << YAML::EndSeq;
    }
    emitter << YAML::EndMap;
    emitter << YAML::EndMap;
    out_file << emitter.c_str();
    out_file.close();

    log_debug(tt::LogFabric, "Serialized inter-mesh port assignment to file: {}", output_file_path.string());
}

void serialize_fabric_debug_manifest_to_file(
    const ControlPlane& control_plane, const std::filesystem::path& output_file_path) {
    using json = nlohmann::ordered_json;

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto fabric_config = control_plane.get_fabric_config();
    const FabricType fabric_type = get_fabric_type(fabric_config, cluster.is_ubb_galaxy());

    json manifest;
    manifest["manifest_version"] = FABRIC_DEBUG_MANIFEST_VERSION;
    manifest["kind"] = "fabric_debug_manifest";

    {
        const auto& distributed_context =
            tt_metal::distributed::multihost::DistributedContext::get_current_world();
        json run;
        run["arch"] = enum_name(cluster.arch());
        run["fabric_config"] = enum_name(fabric_config);
        run["fabric_type"] = enum_name(fabric_type);
        run["reliability_mode"] = enum_name(control_plane.get_fabric_reliability_mode());
        run["tensix_config"] = enum_name(control_plane.get_fabric_tensix_config());
        run["udm_mode"] = enum_name(control_plane.get_fabric_udm_mode());
        run["host_rank"] = *control_plane.get_local_host_rank_id_binding();
        run["mpi_rank"] = *distributed_context->rank();
        run["world_size"] = *distributed_context->size();
        json local_mesh_ids = json::array();
        for (const auto& mesh_id : control_plane.get_local_mesh_id_bindings()) {
            local_mesh_ids.push_back(*mesh_id);
        }
        run["local_mesh_ids"] = std::move(local_mesh_ids);
        manifest["run"] = std::move(run);
    }

    // Chips and their routers describe what to peek; links describe what to draw. Both key on
    // (mesh, chip, channel) so a snapshot can be joined onto either.
    json meshes = json::array();
    json links = json::array();

    auto mesh_ids = mesh_graph.get_all_mesh_ids();
    std::sort(mesh_ids.begin(), mesh_ids.end(), [](const MeshId& lhs, const MeshId& rhs) { return *lhs < *rhs; });

    for (const auto& mesh_id : mesh_ids) {
        const MeshShape mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
        const bool is_2d_mesh = mesh_shape.dims() == 2;

        json mesh;
        mesh["mesh_id"] = *mesh_id;
        json shape = json::array();
        for (size_t dim = 0; dim < mesh_shape.dims(); ++dim) {
            shape.push_back(mesh_shape[dim]);
        }
        mesh["shape"] = std::move(shape);

        // has_genuine_torus_axis() is defined only for a 2D shape, and deliberately reports false for a
        // declared torus axis whose extent is too small to realize a distinct wrap edge.
        if (is_2d_mesh) {
            json torus;
            torus["y"] = has_genuine_torus_axis(fabric_type, mesh_shape, 0);
            torus["x"] = has_genuine_torus_axis(fabric_type, mesh_shape, 1);
            mesh["torus"] = std::move(torus);
        }

        json chips = json::array();
        for (const auto& [_, fabric_chip_id] : mesh_graph.get_chip_ids(mesh_id)) {
            const FabricNodeId node(mesh_id, fabric_chip_id);
            const MeshCoordinate mesh_coord = mesh_graph.chip_to_coordinate(mesh_id, fabric_chip_id);
            const auto physical_chip_id = control_plane.try_get_physical_chip_id_from_fabric_node_id(node);
            // A chip this rank cannot map to a physical device is a chip it cannot peek, so resolvability is
            // the practical definition of locality. Non-local chips still appear, so the viewer can draw the
            // whole mesh and show the host boundary.
            const bool is_local = physical_chip_id.has_value();

            json chip;
            chip["fabric_chip_id"] = fabric_chip_id;
            json coord = json::array();
            for (size_t dim = 0; dim < mesh_coord.dims(); ++dim) {
                coord.push_back(mesh_coord[dim]);
            }
            chip["mesh_coord"] = std::move(coord);
            if (is_local) {
                chip["physical_chip_id"] = *physical_chip_id;
                // Hex string: the value exceeds what JSON numbers represent exactly.
                chip["asic_id"] = fmt::format("0x{:016x}", *control_plane.get_asic_id_from_fabric_node_id(node));
            } else {
                chip["physical_chip_id"] = json(nullptr);
                chip["asic_id"] = json(nullptr);
            }
            chip["is_local"] = is_local;

            json routers = json::array();
            if (is_local) {
                const auto intermesh_chan_list = control_plane.get_intermesh_facing_eth_chans(node);
                const std::set<chan_id_t> intermesh_chans(intermesh_chan_list.begin(), intermesh_chan_list.end());
                const auto& soc_desc = cluster.get_soc_desc(*physical_chip_id);

                // get_active_fabric_eth_channels() is the narrow router set (link up, assigned
                // EthRouterMode::FABRIC_ROUTER, survived routing-plane trimming) and is a std::set keyed on
                // channel, so iteration is already sorted.
                for (const auto& [chan, eth_direction] : control_plane.get_active_fabric_eth_channels(node)) {
                    const RoutingDirection direction = control_plane.eth_direction_to_routing_direction(eth_direction);
                    const char* link_class = intermesh_chans.contains(chan) ? "intermesh" : "intramesh";
                    const auto routing_plane = control_plane.get_routing_plane_id(node, chan);
                    const auto peer = control_plane.try_get_connected_mesh_chip_chan_ids(node, chan);

                    json router;
                    router["eth_chan"] = chan;
                    router["direction"] = enum_name(direction);
                    router["routing_plane"] = routing_plane;
                    router["link_class"] = link_class;
                    const auto logical_core = soc_desc.get_eth_core_for_channel(chan, CoordSystem::LOGICAL);
                    router["logical_core"] = json::array({logical_core.x, logical_core.y});
                    const auto virtual_core = cluster.get_virtual_coordinate_from_logical_coordinates(
                        *physical_chip_id, tt::tt_metal::CoreCoord(logical_core.x, logical_core.y), CoreType::ETH);
                    router["virtual_core"] = json::array({virtual_core.x, virtual_core.y});
                    routers.push_back(std::move(router));

                    // Wrap edges are resolved here rather than inferred by the viewer: a link wraps when its
                    // axis genuinely closes and its coordinate delta spans the mesh.
                    const bool is_east_west =
                        direction == RoutingDirection::E || direction == RoutingDirection::W;
                    const bool is_north_south =
                        direction == RoutingDirection::N || direction == RoutingDirection::S;
                    bool wrap = false;
                    if (is_2d_mesh && peer.has_value() && peer->first.mesh_id == mesh_id &&
                        (is_east_west || is_north_south)) {
                        const uint32_t axis = is_east_west ? 1 : 0;
                        if (has_genuine_torus_axis(fabric_type, mesh_shape, axis)) {
                            const auto peer_coord = mesh_graph.chip_to_coordinate(mesh_id, peer->first.chip_id);
                            const uint32_t here = mesh_coord[axis];
                            const uint32_t there = peer_coord[axis];
                            const uint32_t delta = here > there ? here - there : there - here;
                            wrap = delta == mesh_shape[axis] - 1;
                        }
                    }

                    json link;
                    link["src"] = fabric_debug_endpoint_json(node, chan);
                    link["dst"] =
                        peer.has_value() ? fabric_debug_endpoint_json(peer->first, peer->second) : json(nullptr);
                    link["direction"] = enum_name(direction);
                    link["routing_plane"] = routing_plane;
                    link["link_class"] = link_class;
                    link["wrap"] = wrap;
                    link["cross_host"] = control_plane.is_cross_host_eth_link(*physical_chip_id, chan);
                    links.push_back(std::move(link));
                }
            }
            chip["routers"] = std::move(routers);
            chips.push_back(std::move(chip));
        }
        mesh["chips"] = std::move(chips);
        meshes.push_back(std::move(mesh));
    }

    manifest["meshes"] = std::move(meshes);
    manifest["links"] = std::move(links);

    std::filesystem::create_directories(output_file_path.parent_path());

    std::ofstream out_file(output_file_path);
    if (!out_file.is_open()) {
        TT_THROW("Failed to open output file: {}", output_file_path.string());
    }
    out_file << manifest.dump(2) << std::endl;
    out_file.close();

    log_debug(tt::LogFabric, "Serialized fabric debug manifest to file: {}", output_file_path.string());
}

}  // namespace tt::tt_fabric
