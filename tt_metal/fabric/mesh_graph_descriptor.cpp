// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <assert.hpp>

#include "protobuf/mesh_graph_descriptor.pb.h"
#include "tt-metalium/mesh_graph_descriptor.hpp"
#include "tt-metalium/mesh_coord.hpp"
#include <tt-logger/tt-logger.hpp>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <unistd.h>

using namespace tt::tt_metal::distributed;

namespace tt::tt_fabric {

namespace {

std::string read_file_to_string(const std::filesystem::path &file_path) {
    std::ifstream input(file_path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

uint32_t get_max_dimensions_for_architecture(proto::Architecture arch) {
    switch (arch) {
        case proto::Architecture::WORMHOLE_B0: return 2;
        case proto::Architecture::BLACKHOLE: return 3;
        case proto::Architecture::INVALID_ARCHITECTURE: return 0;
        default: return 0;
    }
}

std::string get_validation_report(const std::vector<std::string>& error_messages) {
    if (error_messages.empty()) {
        return "No validation errors found.\n";
    }

    std::ostringstream report;
    report << "=== MeshGraphDescriptor Validation Report ===\n\n";
    report << "Errors:\n";
    for (const auto& error : error_messages) {
        report << "  - " << error << "\n";
    }
    report << "\n";

    return report.str();
}

LocalNodeId get_device_id(const MeshCoordinate& mesh_coord, const MeshShape& mesh_shape) {
    // Check that mesh_coord is within mesh_shape
    TT_FATAL(mesh_coord[0] < mesh_shape[0] && mesh_coord[1] < mesh_shape[1], "Mesh coordinate {} is out of bounds for mesh shape {}", mesh_coord, mesh_shape);
    return mesh_coord[0] * mesh_shape[1] + mesh_coord[1];
}

std::unordered_map<GlobalNodeId, std::vector<ConnectionData>> get_valid_connections(
        const MeshCoordinate& src_mesh_coord,
        const MeshCoordinateRange& mesh_coord_range,
        const InstanceData& instance) {

    std::unordered_map<GlobalNodeId, std::vector<ConnectionData>> connections;

    const auto* mesh_desc = std::get<const proto::MeshDescriptor*>(instance.desc);
    const auto& topology_types = mesh_desc->device_topology().dim_types();
    const auto& channels_count = mesh_desc->channels().count();
    const auto& policy = mesh_desc->channels().policy();

    MeshShape mesh_shape = mesh_coord_range.shape();
    MeshCoordinate N(src_mesh_coord[0] - 1, src_mesh_coord[1]);
    MeshCoordinate E(src_mesh_coord[0], src_mesh_coord[1] + 1);
    MeshCoordinate S(src_mesh_coord[0] + 1, src_mesh_coord[1]);
    MeshCoordinate W(src_mesh_coord[0], src_mesh_coord[1] - 1);

    if (topology_types[0] == proto::TorusTopology::RING) {
        N = MeshCoordinate((src_mesh_coord[0] - 1 + mesh_shape[0]) % mesh_shape[0], src_mesh_coord[1]);
        S = MeshCoordinate((src_mesh_coord[0] + 1) % mesh_shape[0], src_mesh_coord[1]);
    }
    if (topology_types[1] == proto::TorusTopology::RING) {
        E = MeshCoordinate(src_mesh_coord[0], (src_mesh_coord[1] + 1) % mesh_shape[1]);
        W = MeshCoordinate(src_mesh_coord[0], (src_mesh_coord[1] - 1 + mesh_shape[1]) % mesh_shape[1]);
    }

    for (const auto& [coord, direction] :
         {std::pair{N, proto::RoutingDirection::N},
          std::pair{E, proto::RoutingDirection::E},
          std::pair{S, proto::RoutingDirection::S},
          std::pair{W, proto::RoutingDirection::W}}) {
        if (mesh_coord_range.contains(coord)) {
            const auto src_device_id = instance.sub_instances_local_id_to_global_id.at(get_device_id(src_mesh_coord, mesh_shape));
            const auto dst_device_id = instance.sub_instances_local_id_to_global_id.at(get_device_id(coord, mesh_shape));

            ConnectionData data{
                .nodes = {src_device_id, dst_device_id},
                .count = channels_count,
                .policy = policy,
                .directional = false,
                .parent_instance_id = instance.global_id,
                .routing_direction = direction, // TODO: Remove after MGD 1.0 is deprecated
            };

            connections[src_device_id].push_back(data);
        }
    }

    return connections;
}

}  // namespace

MeshGraphDescriptor::MeshGraphDescriptor(const std::string& text_proto, const bool backwards_compatible) : top_level_id_(static_cast<GlobalNodeId>(-1)), backwards_compatible_(backwards_compatible) {
    proto::MeshGraphDescriptor temp_proto;
    google::protobuf::TextFormat::Parser parser;

    // Allowing for back and forward compatibility for fields not currently in the proto file
    parser.AllowUnknownField(true);
    parser.AllowUnknownExtension(true);

    TT_FATAL(parser.ParseFromString(text_proto, &temp_proto), "Failed to parse MeshGraphDescriptor textproto");

    // Set defaults for missing fields
    set_defaults(temp_proto);

    // Validate the proto
    const auto errors = static_validate(temp_proto, backwards_compatible);
    TT_FATAL(errors.empty(), "Failed to validate MeshGraphDescriptor textproto: \n{}", get_validation_report(errors));

    proto_ = std::make_unique<proto::MeshGraphDescriptor>(temp_proto);

    populate();
}

MeshGraphDescriptor::MeshGraphDescriptor(const std::filesystem::path& text_proto_file_path, const bool backwards_compatible) :
    MeshGraphDescriptor(read_file_to_string(text_proto_file_path.string()), backwards_compatible) {}

MeshGraphDescriptor::~MeshGraphDescriptor() = default;

proto::Architecture MeshGraphDescriptor::get_arch() const {
    // All meshes must have the same arch
    return proto_->mesh_descriptors(0).arch();
}

uint32_t MeshGraphDescriptor::get_num_eth_ports_per_direction() const {
    return proto_->mesh_descriptors(0).channels().count();
}

void MeshGraphDescriptor::set_defaults(proto::MeshGraphDescriptor& proto) {
    // Set the default for channel policy to strict if not specified
    for (auto& mesh : *proto.mutable_mesh_descriptors()) {
        if (mesh.has_channels() && !mesh.channels().has_policy()) {
            mesh.mutable_channels()->set_policy(proto::Policy::STRICT);
        }
    }

    for (auto& graph : *proto.mutable_graph_descriptors()) {
        // Set default policy for graph topology channels
        if (graph.has_graph_topology() && graph.graph_topology().has_channels() &&
            !graph.graph_topology().channels().has_policy()) {
            graph.mutable_graph_topology()->mutable_channels()->set_policy(proto::Policy::STRICT);
        }

        // Set default policy for connection channels
        for (auto& connection : *graph.mutable_connections()) {
            if (connection.has_channels() && !connection.channels().has_policy()) {
                connection.mutable_channels()->set_policy(proto::Policy::STRICT);
            }
        }
    }

    // Set dim_types to LINE if not specified for each dimension
    for (auto& mesh : *proto.mutable_mesh_descriptors()) {
        if (mesh.device_topology().dim_types_size() < mesh.device_topology().dims_size()) {
            for (int i = mesh.device_topology().dim_types_size(); i < mesh.device_topology().dims_size(); i++) {
                mesh.mutable_device_topology()->mutable_dim_types()->Add(proto::TorusTopology::LINE);
            }
        }
    }
}

std::vector<std::string> MeshGraphDescriptor::static_validate(const proto::MeshGraphDescriptor& proto, const bool backwards_compatible) {
    std::vector<std::string> all_errors;

    // Run validation groups with early exit checkpoints
    {
        validate_basic_structure(proto, all_errors);
        if (!all_errors.empty()) return all_errors;
    }

    {
        validate_names(proto, all_errors);
        validate_channels(proto, all_errors);
        validate_architecture_consistency(proto, all_errors);
        if (!all_errors.empty()) return all_errors;
    }

    {
        validate_mesh_topology(proto, all_errors);
        validate_express_connections(proto, all_errors);
        validate_graph_descriptors(proto, all_errors);
        validate_graph_topology_and_connections(proto, all_errors);
        if (!all_errors.empty()) return all_errors;
    }

    {
        if (backwards_compatible) {
            validate_legacy_requirements(proto, all_errors);
        }
        if (!all_errors.empty()) return all_errors;
    }

    return all_errors;
}

void MeshGraphDescriptor::populate() {
    populate_descriptors();

    populate_top_level_instance();

    pre_populate_connections_lookups();

    populate_connections();
}

void MeshGraphDescriptor::populate_top_level_instance() {
    std::vector<GlobalNodeId> hierarchy;
    top_level_id_ = populate_instance(proto_->top_level_instance(), hierarchy);
}

void MeshGraphDescriptor::validate_basic_structure(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& errors) {
    if (proto.mesh_descriptors_size() == 0) {
        errors.push_back("There must be at least one mesh descriptor");
    }
    if (!proto.has_top_level_instance()) {
        errors.push_back("Top level instance is required");
    }
}



void MeshGraphDescriptor::validate_names(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {

    unsigned int mesh_counter = 0;

    // Check that all mesh descriptors have a unique name
    std::unordered_set<std::string> mesh_names;
    for (const auto& mesh : proto.mesh_descriptors()) {
        mesh_counter++;
        if (mesh.name().empty()) {
            error_messages.push_back(
                fmt::format(
                    "Mesh descriptor {} has no name",
                    mesh_counter
                )
            );
            continue;
        }
        auto [it, inserted] = mesh_names.insert(mesh.name());
        if (!inserted) {
            error_messages.push_back(
                fmt::format(
                    "Mesh descriptor name is not unique (Mesh: {})",
                    mesh.name()
                )
            );
        }
    }

    unsigned int graph_counter = 0;

    // Check that all graph descriptors have a unique name
    std::unordered_set<std::string> graph_names;
    for (const auto& graph : proto.graph_descriptors()) {
        graph_counter++;
        if (graph.name().empty()) {
            error_messages.push_back(
                fmt::format(
                    "Graph descriptor {} has no name",
                    graph_counter
                )
            );
            continue;
        }
        auto [it, inserted] = graph_names.insert(graph.name());
        if (!inserted) {
            error_messages.push_back(
                fmt::format(
                    "Graph descriptor name is not unique (Graph: {})",
                    graph.name()
                )
            );
        }

        // TYPE name cannot be DEVICE or MESH
        const auto& type = graph.type();
        if (type == "DEVICE" || type == "MESH") {
            error_messages.push_back(
                fmt::format(
                    "Graph descriptor type cannot be DEVICE or MESH (Graph: {})",
                    graph.name()
                )
            );
        }
    }

}


void MeshGraphDescriptor::validate_mesh_topology(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {

    // Validate basic mesh properties (names and dimensions)
    for (const auto& mesh : proto.mesh_descriptors()) {
        // Check that all dims are positive
        for (const auto& dim : mesh.device_topology().dims()) {
            if (dim <= 0) {
                error_messages.push_back(
                    fmt::format(
                        "Device topology dimensions must be positive (Mesh: {})",
                        mesh.name()
                    )
                );
                continue;
            }
        }

        // Check that device topology dimensions and types are the same size
        if (mesh.device_topology().dim_types_size() > 0) {
            if (mesh.device_topology().dims_size() != mesh.device_topology().dim_types_size()) {
                error_messages.push_back(
                    fmt::format(
                        "Device topology dimensions and types must be the same size (Mesh: {})",
                        mesh.name()
                    )
                );
                continue;
            }
        }

        // Check that the device and host topology dimensions are the same size
        if (mesh.device_topology().dims_size() != mesh.host_topology().dims_size()) {
            error_messages.push_back(
                fmt::format(
                    "Device and host topology dimensions must be the same size (Mesh: {})",
                    mesh.name()
                )
            );
            continue;
        }

        // Check that the device topology dimensions are divisible by the host topology dimensions
        if (mesh.device_topology().dims_size() > 0) {
            for (int i = 0; i < mesh.device_topology().dims_size(); i++) {
                if (mesh.device_topology().dims(i) % mesh.host_topology().dims(i) != 0) {
                    error_messages.push_back(
                        fmt::format(
                            "Device topology dimensions must be divisible by host topology dimensions (Mesh: {})",
                            mesh.name()
                        )
                    );
                    continue;
                }
            }
        }
    }

}

void MeshGraphDescriptor::validate_architecture_consistency(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {

    // Check all architectures are the same
    if (proto.mesh_descriptors_size() > 0) {
        proto::Architecture first_arch = proto.mesh_descriptors(0).arch();
        if (!std::all_of(proto.mesh_descriptors().begin(), proto.mesh_descriptors().end(),
                        [first_arch](const auto& mesh) { return mesh.arch() == first_arch; })) {
            error_messages.push_back("All mesh descriptors must have the same architecture");
            return;
        }
    }

// Verify that arch, device and host topology must exist in mesh descriptors
    for (const auto& mesh : proto.mesh_descriptors()) {
        if (mesh.arch() == proto::Architecture::INVALID_ARCHITECTURE) {
            error_messages.push_back(
                fmt::format(
                    "Mesh descriptor must have a valid architecture (Mesh: {})",
                    mesh.name()
                )
            );
            continue;
        }

        // Validate architecture and dimension limits
        const uint32_t max_num_dims = get_max_dimensions_for_architecture(mesh.arch());
        if (max_num_dims == 0) {
            error_messages.push_back(
                fmt::format(
                    "Invalid architecture (Mesh: {})",
                    mesh.name()
                )
            );
            continue;
        }

        // Check that the number of dimensions is not greater than the maximum allowed for the architecture
        if (mesh.device_topology().dims_size() > max_num_dims) {
            error_messages.push_back(
                fmt::format(
                    "Architecture devices allow a maximum of {} dimensions, but {} were provided (Mesh: {})",
                    max_num_dims,
                    mesh.device_topology().dims_size(),
                    mesh.name()
                )
            );
            continue;
        }
    }

}

void MeshGraphDescriptor::validate_channels(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {

    // Check all channel counts > 0
    for (const auto& mesh : proto.mesh_descriptors()) {
        if (mesh.channels().count() <= 0) {
            error_messages.push_back(
                fmt::format(
                    "Channel count must be positive (Mesh: {})",
                    mesh.name()
                )
            );
        }
    }

    // Check that channels in graph topology are positive
    for (const auto& graph : proto.graph_descriptors()) {
        if (graph.has_graph_topology() && graph.graph_topology().channels().count() <= 0) {
            error_messages.push_back(
                fmt::format(
                    "Graph topology channel count must be positive (Graph: {})",
                    graph.name()
                )
            );
        }
    }

    // Check all channel counts > 0 in graph descriptors and connections
    for (const auto& graph : proto.graph_descriptors()) {
        // Check connection-level channels and validate connection nodes
        for (const auto& connection : graph.connections()) {
            if (connection.channels().count() <= 0) {
                error_messages.push_back(
                    fmt::format(
                        "Connection channel count must be positive (Graph: {})",
                        graph.name()
                    )
                );
            }
        }
    }

}

void MeshGraphDescriptor::validate_express_connections(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {

    // Validate express connections
    for (const auto& mesh : proto.mesh_descriptors()) {
        const uint32_t num_dims = mesh.device_topology().dims_size();

        // Calculate the number of devices in the mesh
        uint32_t num_devices = 1;
        for (uint32_t i = 0; i < num_dims; i++) {
            num_devices *= mesh.device_topology().dims(i);
        }

        // Check that express connections are valid and have the right number of devices
        for (const auto& express_connection : mesh.express_connections()) {
            if (express_connection.src() < 0 || express_connection.src() >= num_devices) {
                error_messages.push_back(
                    fmt::format(
                        "Express connection source is out of bounds (Mesh: {})",
                        mesh.name()
                    )
                );
            }
            if (express_connection.dst() < 0 || express_connection.dst() >= num_devices) {
                error_messages.push_back(
                    fmt::format(
                        "Express connection destination is out of bounds (Mesh: {})",
                        mesh.name()
                    )
                );
            }
        }
    }

}

void MeshGraphDescriptor::validate_graph_descriptors(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {

    // Check that there is at least one instance in the graph and validate references
    for (const auto& graph : proto.graph_descriptors()) {
        if (graph.instances_size() == 0) {
            error_messages.push_back(
                fmt::format(
                    "Graph descriptor must have at least one instance (Graph: {})",
                    graph.name()
                )
            );
        }
    }

    // Verify that type is set in graph descriptors
    for (const auto& graph : proto.graph_descriptors()) {
        if (graph.type().empty()) {
            error_messages.push_back(
                fmt::format(
                    "Graph descriptor must have a type specified (Graph: {})",
                    graph.name()
                )
            );
        }
    }

}

void MeshGraphDescriptor::validate_graph_topology_and_connections(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {

    // Combine all checks into a single loop over graph_descriptors
    for (const auto& graph : proto.graph_descriptors()) {
        // Check that both graph_topology and connections are not defined at the same time
        if (graph.has_graph_topology() && graph.connections_size() > 0) {
            error_messages.push_back(
                fmt::format(
                    "Graph descriptor cannot have both graph_topology and connections defined (Graph: {})",
                    graph.name()
                )
            );
            continue;
        }

        // Check connections have at least 2 nodes
        for (const auto& connection : graph.connections()) {
            if (connection.nodes_size() < 2) {
                error_messages.push_back(
                    fmt::format(
                        "Connection must have at least two nodes (Graph: {})",
                        graph.name()
                    )
                );
            }
        }
    }

}

void MeshGraphDescriptor::validate_legacy_requirements(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages) {
    // Validate that channels count must all be exactly the same
    const uint32_t first_channels_count = proto.mesh_descriptors(0).channels().count();
    for (const auto& mesh : proto.mesh_descriptors()) {
        if (mesh.channels().count() != first_channels_count) {
            error_messages.push_back(
                fmt::format( "MGD 1.0 Compatibility requirement: Channel count must all be exactly the same (Mesh: {})", mesh.name()
            ));
        }
    }

    // Check that there are only 2 dimensions in the device topology and host topology
    for (const auto& mesh : proto.mesh_descriptors()) {
        if (mesh.device_topology().dims_size() != 2 || mesh.host_topology().dims_size() != 2) {
            error_messages.push_back(fmt::format(
                "MGD 1.0 Compatibility requirement: There can only be 2 dimensions in the device topology and host "
                "topology (Mesh: {})",
                mesh.name()));
        }
    }

    // Check that there is only a FABRIC level graph
    if (proto.graph_descriptors_size() > 1) {
        error_messages.push_back(fmt::format(
            "MGD 1.0 Compatibility requirement: There can only be one FABRIC level graph or less"
        ));
    }

    for (const auto& graph : proto.graph_descriptors()) {
        if (graph.type() != "FABRIC") {
            error_messages.push_back(fmt::format(
                "MGD 1.0 Compatibility requirement: There can only be one FABRIC level graph (Graph: {})",
                graph.name()));
        }
    }

    // Connections have to be specific down to the device level
    for (const auto& graph : proto.graph_descriptors()) {
        for (const auto& connection : graph.connections()) {
            for (const auto& node : connection.nodes()) {
                if (!node.mesh().has_device_id()) {
                    error_messages.push_back(
                        fmt::format( "MGD 1.0 Compatibility requirement: Connections have to be specific down to the device level (Graph: {})", graph.name())
                    );
                }
            }
        }
    }

    // Disable graph layout topologies for now
    for (const auto& graph : proto.graph_descriptors()) {
        if (graph.has_graph_topology()) {
            error_messages.push_back(
                fmt::format( "MGD 1.0 Compatibility requirement: Graph layout topologies are not supported (Graph: {})", graph.name())
            );
        }
    }

    // Check that the directions are set properly
    for (const auto& graph : proto.graph_descriptors()) {
        for (const auto& connection : graph.connections()) {
            if (connection.routing_direction_size() != connection.nodes_size()) {
                error_messages.push_back(fmt::format(
                    "MGD 1.0 Compatibility requirement: Routing direction must have the same number of nodes (Graph: "
                    "{})",
                    graph.name()));
            }
            for (const auto& direction : connection.routing_direction()) {
                if (direction == proto::RoutingDirection::INVALID) {
                    error_messages.push_back(fmt::format(
                        "MGD 1.0 Compatibility requirement: Routing direction must be valid (Graph: {})",
                        graph.name()));
                }
            }
        }
    }

    // Check that connections only have 2 nodes
    for (const auto& graph : proto.graph_descriptors()) {
        for (const auto& connection : graph.connections()) {
            if (connection.nodes_size() != 2) {
                error_messages.push_back(fmt::format(
                    "MGD 1.0 Compatibility requirement: Connections must have exactly 2 nodes (Graph: {})",
                    graph.name()));
            }
        }
    }
}

void MeshGraphDescriptor::populate_descriptors() {
    mesh_desc_by_name_.clear();
    graph_desc_by_name_.clear();
    mesh_desc_by_name_.reserve(proto_->mesh_descriptors_size());
    graph_desc_by_name_.reserve(proto_->graph_descriptors_size());
    // Use string_view into proto_ storage; safe as long as proto_ outlives maps
    for (int i = 0; i < proto_->mesh_descriptors_size(); ++i) {
        const auto& mesh = proto_->mesh_descriptors(i);
        mesh_desc_by_name_.emplace(mesh.name(), &mesh);
    }
    for (int i = 0; i < proto_->graph_descriptors_size(); ++i) {
        const auto& graph = proto_->graph_descriptors(i);
        graph_desc_by_name_.emplace(graph.name(), &graph);
    }
}

GlobalNodeId MeshGraphDescriptor::populate_instance(
    const proto::NodeRef& node_ref, std::vector<GlobalNodeId>& hierarchy) {
    GlobalNodeId global_id;
    if (node_ref.has_mesh()) {
        global_id = populate_mesh_instance(node_ref.mesh(), hierarchy);
    } else if (node_ref.has_graph()) {
        global_id = populate_graph_instance(node_ref.graph(), hierarchy);
    } else {
        TT_THROW("Invalid NodeRef: neither mesh nor graph set");
        return -1;
    }

    auto & instance = instances_.at(global_id);

    // Check that graph descriptor type is not already in the hierarchy
    for (const auto& id : hierarchy) {
        auto & instance_in_hierarchy = instances_.at(id);
        TT_FATAL(instance_in_hierarchy.type != instance.type, "Graph descriptor type {} already exists in hierarchy", instance.type);
    }

    return global_id;
}

GlobalNodeId MeshGraphDescriptor::populate_mesh_instance(
    const proto::MeshRef& mesh_ref, std::vector<GlobalNodeId>& hierarchy) {
    const std::string & descriptor_name = mesh_ref.mesh_descriptor();
    const auto it = mesh_desc_by_name_.find(descriptor_name);
    TT_FATAL(it != mesh_desc_by_name_.end(), "Mesh descriptor {} not found in instance", descriptor_name);
    const auto* mesh_desc = it->second;

    InstanceData data{
        .local_id = mesh_ref.mesh_id(),
        .name = mesh_desc->name(),
        .type = "MESH",
        .kind = NodeKind::Mesh,
        .desc = mesh_desc,
    };

    const auto & [instance_it, _] = instances_.emplace(data.global_id, std::move(data));
    auto & instance = instance_it->second;

    instance.hierarchy = hierarchy;

    // Populate devices in the mesh
    uint32_t num_devices = 1;
    for (const auto& dim : mesh_desc->device_topology().dims()) {
        num_devices *= dim;
    }

    instance.sub_instances.reserve(num_devices);
    instance.sub_instances_local_id_to_global_id.reserve(num_devices);

    hierarchy.push_back(instance.global_id);
    for (LocalNodeId i = 0; i < num_devices; ++i) {
        const auto device_id = populate_device_instance(i, hierarchy);

        instance.sub_instances.insert(device_id);
        instance.sub_instances_local_id_to_global_id.emplace(i, device_id);
    }
    hierarchy.pop_back();

    add_to_fast_lookups(instance);

    return instance.global_id;
}

GlobalNodeId MeshGraphDescriptor::populate_device_instance(LocalNodeId local_id, std::vector<GlobalNodeId>& hierarchy) {
    const std::string name = "D" + std::to_string(local_id);
    InstanceData data{
        .local_id = local_id,
        .name = name,
        .type = "DEVICE",
        .kind = NodeKind::Device,
    };
    const auto global_id = data.global_id;
    instances_.emplace(global_id, std::move(data));
    const auto it_instance = instances_.find(global_id);
    TT_FATAL(it_instance != instances_.end(), "Failed to create device instance for global id {}", global_id);
    auto & instance = it_instance->second;

    instance.hierarchy = hierarchy;

    device_instances_.push_back(instance.global_id);
    instances_by_type_["DEVICE"].push_back(instance.global_id);
    // Use stable storage for key to avoid dangling string_view
    instances_by_name_[instance.name].push_back(instance.global_id);

    return instance.global_id;
}

GlobalNodeId MeshGraphDescriptor::populate_graph_instance(
    const proto::GraphRef& graph_ref, std::vector<GlobalNodeId>& hierarchy) {
    const std::string & descriptor_name = graph_ref.graph_descriptor();
    const auto it = graph_desc_by_name_.find(descriptor_name);
    TT_FATAL(it != graph_desc_by_name_.end(), "Graph descriptor {} not found in instance", descriptor_name);
    const auto* graph_desc = it->second;

    InstanceData data{
        .local_id = static_cast<LocalNodeId>(graph_ref.graph_id()),
        .name = graph_desc->name(),
        .type = graph_desc->type(),
        .kind = NodeKind::Graph,
        .desc = graph_desc,
    };

    const auto emplace_result = instances_.emplace(data.global_id, std::move(data));
    auto & instance = emplace_result.first->second;

    instance.hierarchy = hierarchy;

    // Populate sub-instances from the graph descriptor
    std::unordered_set<GlobalNodeId> children_global_ids;
    children_global_ids.reserve(graph_desc->instances_size());
    instance.sub_instances_local_id_to_global_id.reserve(graph_desc->instances_size());

    hierarchy.push_back(instance.global_id);
    std::string_view child_graph_type;
    for (const auto& sub_ref : graph_desc->instances()) {
        GlobalNodeId child = populate_instance(sub_ref, hierarchy);

        const auto it_child = instances_.find(child);
        TT_FATAL(it_child != instances_.end(), "Child instance id {} not found while populating graph instance", child);
        const auto & child_instance = it_child->second;

        // Check that the child instance created has the same type as rest of the graph descriptor
        if (child_instance.kind == NodeKind::Graph) {
            if (child_graph_type.empty()) {
                child_graph_type = child_instance.type.c_str();
            } else {
                TT_FATAL(child_graph_type == child_instance.type, "Graph instance type {} does not match graph descriptor child type {}", std::string(child_graph_type), std::string(child_instance.type));
            }
        }

        TT_FATAL(!instance.sub_instances_local_id_to_global_id.contains(child_instance.local_id), "Graph instance id {} already exists in this graph", child_instance.local_id);

        children_global_ids.insert(child_instance.global_id);
        instance.sub_instances_local_id_to_global_id.emplace(child_instance.local_id, child_instance.global_id);
    }
    hierarchy.pop_back();

    instance.sub_instances = std::move(children_global_ids);

    graph_instances_.push_back(instance.global_id);
    instances_by_type_[graph_desc->type()].push_back(instance.global_id);
    instances_by_name_[graph_desc->name()].push_back(instance.global_id);

    return instance.global_id;
}

void MeshGraphDescriptor::populate_connections() {
    for (const auto & mesh_id : mesh_instances_) {
        populate_intra_mesh_connections(mesh_id);
        populate_intra_mesh_express_connections(mesh_id);
    }
    for (const auto & graph_id : graph_instances_) {
        populate_inter_mesh_connections(graph_id);
    }
}

void MeshGraphDescriptor::add_to_fast_lookups(const InstanceData& instance) {
    // Add to type-based lookup
    instances_by_type_[instance.type].push_back(instance.global_id);

    // Add to name-based lookup
    instances_by_name_[instance.name].push_back(instance.global_id);

    // Add to kind-specific lookups
    switch (instance.kind) {
        case NodeKind::Mesh:
            mesh_instances_.push_back(instance.global_id);
            break;
        case NodeKind::Graph:
            graph_instances_.push_back(instance.global_id);
            break;
        case NodeKind::Device:
            device_instances_.push_back(instance.global_id);
            break;
    }
}

void MeshGraphDescriptor::pre_populate_connections_lookups() {
    for (const auto& [instance_id, instance] : instances_) {
    // Add empty vectors for the instance's type, instance id, and source device id
    if (connections_by_type_.find(instance.type) == connections_by_type_.end()) {
            connections_by_type_.emplace(instance.type, std::vector<ConnectionId>());
        }
        if (connections_by_instance_id_.find(instance_id) == connections_by_instance_id_.end()) {
            connections_by_instance_id_.emplace(instance.global_id, std::vector<ConnectionId>());
        }
        if (connections_by_source_device_id_.find(instance_id) == connections_by_source_device_id_.end()) {
            connections_by_source_device_id_.emplace(instance.global_id, std::vector<ConnectionId>());
        }
    }

    // TODO: Remove this after MGD 1.0 is deprecated
    if (connections_by_type_.find("FABRIC") == connections_by_type_.end()) {
        connections_by_type_.emplace("FABRIC", std::vector<ConnectionId>());
    }
}

void MeshGraphDescriptor::add_connection_to_fast_lookups(const ConnectionData& connection, const std::string& type) {
    // Add to instance-based lookup
    connections_by_instance_id_[connection.parent_instance_id].push_back(connection.connection_id);

    // Add to type-based lookup
    connections_by_type_[type].push_back(connection.connection_id);

    // Add to source device lookup
    if (!connection.nodes.empty()) {
        connections_by_source_device_id_[connection.nodes[0]].push_back(connection.connection_id);
    }
}

void MeshGraphDescriptor::populate_intra_mesh_connections(GlobalNodeId mesh_id) {

    auto & instance = instances_.at(mesh_id);

    const auto mesh_desc = std::get<const proto::MeshDescriptor*>(instance.desc);

    TT_FATAL(mesh_desc->device_topology().dims_size() == 2, "MGD currently only supports 2D meshes");

    // TODO: Expand this for 2+ dimensional meshes
    const std::uint32_t mesh_ns_size = mesh_desc->device_topology().dims(0);
    const std::uint32_t mesh_ew_size = mesh_desc->device_topology().dims(1);
    const auto mesh_shape = MeshShape(mesh_ns_size, mesh_ew_size);

    for (const auto& src_mesh_coord : MeshCoordinateRange(mesh_shape)) {
        const auto connections = get_valid_connections(src_mesh_coord, MeshCoordinateRange(mesh_shape), instance);

        for (const auto& [src_device_id, per_source_connections] : connections) {
            for (const auto& connection_data : per_source_connections) {
                const auto id = connection_data.connection_id;
                add_connection_to_fast_lookups(connection_data, instance.type);
                connections_.emplace(id, connection_data);
            }
        }
    }
}

void MeshGraphDescriptor::populate_intra_mesh_express_connections(GlobalNodeId mesh_id) {
    auto & instance = instances_.at(mesh_id);
    const auto mesh_desc = std::get<const proto::MeshDescriptor*>(instance.desc);
    for (const auto& express_connection : mesh_desc->express_connections()) {
        const auto src_device_id = instance.sub_instances_local_id_to_global_id.at(express_connection.src());
        const auto dst_device_id = instance.sub_instances_local_id_to_global_id.at(express_connection.dst());

        ConnectionData data{
            .nodes = {src_device_id, dst_device_id},
            .count = mesh_desc->channels().count(),
            .policy = mesh_desc->channels().policy(),
            .directional = false,
            .parent_instance_id = mesh_id,
            .routing_direction = proto::RoutingDirection::C,  // TODO: Remove after MGD 1.0 is deprecated
        };

        add_connection_to_fast_lookups(data, instance.type);
        connections_.emplace(data.connection_id, std::move(data));

        ConnectionData data_reverse{
            .nodes = {dst_device_id, src_device_id},
            .count = mesh_desc->channels().count(),
            .policy = mesh_desc->channels().policy(),
            .directional = false,
            .parent_instance_id = mesh_id,
            .routing_direction = proto::RoutingDirection::C,  // TODO: Remove after MGD 1.0 is deprecated
        };

        add_connection_to_fast_lookups(data_reverse, instance.type);
        connections_.emplace(data_reverse.connection_id, std::move(data_reverse));
    }
}

GlobalNodeId MeshGraphDescriptor::find_instance_by_ref(
    GlobalNodeId parent_instance_id, const proto::NodeRef& node_ref) {
    auto & parent_instance = instances_.at(parent_instance_id);

    if (node_ref.has_mesh()) {
        // Check the instance id exists References are indexed by local id
        const auto local_instance_id = node_ref.mesh().mesh_id();
        const auto it2 = parent_instance.sub_instances_local_id_to_global_id.find(local_instance_id);
        TT_FATAL(it2 != parent_instance.sub_instances_local_id_to_global_id.end(), "Mesh instance id {} not found in parent instance", local_instance_id);

        const auto global_instance_id = it2->second;
        auto & referenced_instance = instances_.at(global_instance_id);

        // Check if the mesh descriptor already exists
        const auto descriptor_name = node_ref.mesh().mesh_descriptor();
        TT_FATAL(descriptor_name == referenced_instance.name, "Mesh descriptor {} does not match referenced instance {}", descriptor_name, referenced_instance.name);

        // Check sub instance exists
        if (node_ref.mesh().has_device_id()) {
            const auto device_id = node_ref.mesh().device_id();
            auto & mesh_instance = instances_.at(global_instance_id);
            const auto it = mesh_instance.sub_instances_local_id_to_global_id.find(device_id);
            TT_FATAL(it != mesh_instance.sub_instances_local_id_to_global_id.end(), "Device id {} not found in mesh instance", device_id);
            return it->second;
        }

        return global_instance_id;

    } else if (node_ref.has_graph()) {
        const auto instance_id = node_ref.graph().graph_id();
        const auto it = parent_instance.sub_instances_local_id_to_global_id.find(instance_id);
        TT_FATAL(it != parent_instance.sub_instances_local_id_to_global_id.end(), "Graph instance id {} not found in parent instance", instance_id);

        const auto global_instance_id = it->second;
        auto & referenced_instance = instances_.at(global_instance_id);

        const auto descriptor_name = node_ref.graph().graph_descriptor();
        TT_FATAL(descriptor_name == referenced_instance.name, "Graph descriptor {} does not match referenced instance {}", descriptor_name, referenced_instance.name);

        if (node_ref.graph().has_sub_ref()) {
            return find_instance_by_ref(global_instance_id, node_ref.graph().sub_ref());
        }

        return global_instance_id;

    }
    TT_THROW("Invalid NodeRef: neither mesh nor graph set");
    return -1;
}

void MeshGraphDescriptor::populate_inter_mesh_connections(GlobalNodeId graph_id) {
    populate_inter_mesh_manual_connections(graph_id);
    populate_inter_mesh_topology_connections(graph_id);
}

void MeshGraphDescriptor::populate_inter_mesh_manual_connections(GlobalNodeId graph_id) {
    auto & instance = instances_.at(graph_id);

    const auto graph_desc = std::get<const proto::GraphDescriptor*>(instance.desc);

    TT_FATAL(graph_desc, "Graph descriptor not found for graph instance {}", graph_id);

    for (const auto& connection : graph_desc->connections()) {

        std::string_view type;

        std::vector<GlobalNodeId> nodes;

        for (const auto& node : connection.nodes()) {
            // Find the referenced instance
            GlobalNodeId ref_instance_id = find_instance_by_ref(graph_id, node);
            auto & ref_instance = instances_.at(ref_instance_id);

            // Check that the referenced instances have the same type
            if (type.empty()) {
                type = ref_instance.type;
            } else {
                TT_FATAL(type == ref_instance.type, "Graph descriptor {} connections must reference instances within same type", instance.name);
            }

            nodes.push_back(ref_instance_id);
        }

        TT_ASSERT(nodes.size() >= 2, "Graph descriptor connections must have at least two nodes");

        // Add the connection in every direction of the connection
        for (std::size_t i = 0; i < connection.nodes_size(); ++i) {
            // Create a copy of the nodes vector and swap the first and i-th elements so source is always first
            std::vector<GlobalNodeId> nodes_copy = nodes;
            std::swap(nodes_copy[0], nodes_copy[i]);

            proto::RoutingDirection routing_direction = proto::RoutingDirection::NONE;
            if (connection.routing_direction_size() != 0) {
                routing_direction = connection.routing_direction(i);
            }

            ConnectionData data{
                .nodes = nodes_copy,
                .count = connection.channels().count(),
                .policy = connection.channels().policy(),
                .directional = connection.directional(),
                .parent_instance_id = graph_id,
                .routing_direction = routing_direction,
            };

            add_connection_to_fast_lookups(data, instance.type);
            connections_.emplace(data.connection_id, std::move(data));

            if (connection.directional()) {
                break;
            }
        }
    }
}

void MeshGraphDescriptor::populate_inter_mesh_topology_connections(GlobalNodeId graph_id) {
    // TODO: This is to be implemented in seperate PR
}


void MeshGraphDescriptor::print_node(GlobalNodeId id, int indent_level) {
    std::string indent(indent_level * 2, ' ');
    std::stringstream ss;

    const auto it = instances_.find(id);
    if (it == instances_.end()) {
        ss << indent << "Unknown instance id: " << id << std::endl;
        log_debug(tt::LogFabric, "{}", ss.str());
        return;
    }

    const InstanceData & inst = it->second;
    if (inst.kind == NodeKind::Mesh) {
        ss << indent << "=== MESH INSTANCE ===" << std::endl;
        ss << indent << "Global ID: " << id << std::endl;
        ss << indent << "Local ID: " << inst.local_id << std::endl;
        ss << indent << "Name: " << inst.name << std::endl;
        const auto* mesh_desc = std::get<const proto::MeshDescriptor*>(inst.desc);
        ss << indent << "Device Topology Dimensions: [";
        for (int i = 0; i < mesh_desc->device_topology().dims_size(); ++i) {
            if (i > 0) ss << ", ";
            ss << mesh_desc->device_topology().dims(i);
        }
        ss << "]" << std::endl;
        ss << indent << "Host Topology Dimensions: [";
        for (int i = 0; i < mesh_desc->host_topology().dims_size(); ++i) {
            if (i > 0) ss << ", ";
            ss << mesh_desc->host_topology().dims(i);
        }
        ss << "]" << std::endl;
        ss << indent << "Channel Count: " << mesh_desc->channels().count() << std::endl;
        ss << indent << "Express Connections: " << mesh_desc->express_connections_size() << std::endl;
        if (!inst.sub_instances.empty()) {
            ss << indent << "Devices:" << std::endl;
            log_debug(tt::LogFabric, "{}", ss.str());
            ss.str(std::string());
            // Print devices in ascending local_id order
            std::vector<std::pair<LocalNodeId, GlobalNodeId>> ordered;
            ordered.reserve(inst.sub_instances_local_id_to_global_id.size());
            for (const auto & kv : inst.sub_instances_local_id_to_global_id) {
                ordered.emplace_back(kv.first, kv.second);
            }
            std::sort(ordered.begin(), ordered.end(), [](const auto & a, const auto & b){ return a.first < b.first; });
            for (const auto & kv : ordered) {
                print_node(kv.second, indent_level + 1);
            }
            return; // children already printed with their own trailing separators
        }
    } else if (inst.kind == NodeKind::Graph) {
        ss << indent << "=== GRAPH INSTANCE ===" << std::endl;
        ss << indent << "Global ID: " << id << std::endl;
        ss << indent << "Local ID: " << inst.local_id << std::endl;
        ss << indent << "Name: " << inst.name << std::endl;
        ss << indent << "Type: " << inst.type << std::endl;
        const auto* graph_desc = std::get<const proto::GraphDescriptor*>(inst.desc);
        ss << indent << "Total Instances in Descriptor: " << graph_desc->instances_size() << std::endl;
        ss << indent << "Connections: " << graph_desc->connections_size() << std::endl;
        if (graph_desc->has_graph_topology()) {
            ss << indent << "Has Graph Topology: Yes" << std::endl;
        }
        if (!inst.sub_instances.empty()) {
            ss << indent << "Sub-instances:" << std::endl;
            log_debug(tt::LogFabric, "{}", ss.str());
            ss.str(std::string());
            // Print children in ascending local_id order
            std::vector<std::pair<LocalNodeId, GlobalNodeId>> ordered;
            ordered.reserve(inst.sub_instances_local_id_to_global_id.size());
            for (const auto & kv : inst.sub_instances_local_id_to_global_id) {
                ordered.emplace_back(kv.first, kv.second);
            }
            std::sort(ordered.begin(), ordered.end(), [](const auto & a, const auto & b){ return a.first < b.first; });
            for (const auto & kv : ordered) {
                print_node(kv.second, indent_level + 1);
            }
            return; // children already printed with their own trailing separators
        }
    } else if (inst.kind == NodeKind::Device) {
        ss << indent << "=== DEVICE INSTANCE ===" << std::endl;
        ss << indent << "Global ID: " << id << std::endl;
        ss << indent << "Local ID: " << inst.local_id << std::endl;
        ss << indent << "Name: " << inst.name << std::endl;
        ss << indent << "Hierarchy Depth: " << inst.hierarchy.size() << std::endl;
    } else {
        ss << indent << "=== UNKNOWN NODE TYPE ===" << std::endl;
        ss << indent << "Global ID: " << id << std::endl;
    }

    ss << indent << "---" << std::endl;
    log_debug(tt::LogFabric, "{}", ss.str());
}

void MeshGraphDescriptor::print_all_nodes() {
    std::stringstream ss;
    ss << "\n=== PRINTING ALL NODE INSTANCES (recursive from top-level) ===" << std::endl;
    ss << "Total instances: " << instances_.size() << std::endl;
    ss << "=====================================" << std::endl;
    log_debug(tt::LogFabric, "{}", ss.str());

    // Start from top-level and recursively print in local-id order
    print_node(top_level_id_, 0);
}
}  // namespace tt::tt_fabric
