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
#include "assert.hpp"

#include "protobuf/mesh_graph_descriptor.pb.h"
#include "tt-metalium/mesh_graph_descriptor.hpp"
#include "tt-metalium/mesh_graph_new.hpp"
#include <tt-logger/tt-logger.hpp>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <unistd.h>

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
}  // namespace

MeshGraphDescriptor::MeshGraphDescriptor(const std::string& text_proto) : top_level_id_(static_cast<NodeId>(-1)) {
    proto::MeshGraphDescriptor temp_proto;
    google::protobuf::TextFormat::Parser parser;

    // Allowing for back and forward compatibility for fields not currently in the proto file
    parser.AllowUnknownField(true);
    parser.AllowUnknownExtension(true);

    TT_FATAL(parser.ParseFromString(text_proto, &temp_proto), "Failed to parse MeshGraphDescriptor textproto");

    // Set defaults for missing fields
    set_defaults(temp_proto);

    // Validate the proto
    auto errors = static_validate(temp_proto);
    TT_FATAL(errors.empty(), "Failed to validate MeshGraphDescriptor textproto: \n{}", get_validation_report(errors));

    proto_ = std::make_unique<proto::MeshGraphDescriptor>(temp_proto);

    populate();
}

MeshGraphDescriptor::MeshGraphDescriptor(const std::filesystem::path& text_proto_file_path) :
    MeshGraphDescriptor(read_file_to_string(text_proto_file_path.string())) {}

MeshGraphDescriptor::~MeshGraphDescriptor() = default;

void MeshGraphDescriptor::set_defaults(proto::MeshGraphDescriptor& proto) {
    // Set the default for channel policy to strict if not specified
    for (auto& mesh : *proto.mutable_mesh_descriptors()) {
        if (mesh.has_channels() && !mesh.channels().has_policy()) {
            mesh.mutable_channels()->set_policy(proto::Policy::STRICT);
        }
    }
    
    for (auto& graph : *proto.mutable_graph_descriptors()) {
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

std::vector<std::string> MeshGraphDescriptor::static_validate(const proto::MeshGraphDescriptor& proto) {
    std::vector<std::string> all_errors;

    // Run validation groups with early exit checkpoints
    {
        auto errs = validate_basic_structure(proto);
        all_errors.insert(all_errors.end(), errs.begin(), errs.end());
        if (!all_errors.empty()) return all_errors;
    }

    {
        auto errs = validate_names(proto);
        all_errors.insert(all_errors.end(), errs.begin(), errs.end());
        auto errs2 = validate_channels(proto);
        all_errors.insert(all_errors.end(), errs2.begin(), errs2.end());
        auto errs3 = validate_architecture_consistency(proto);
        all_errors.insert(all_errors.end(), errs3.begin(), errs3.end());
        if (!all_errors.empty()) return all_errors;
    }

    {
        auto errs = validate_mesh_topology(proto);
        all_errors.insert(all_errors.end(), errs.begin(), errs.end());
        auto errs2 = validate_express_connections(proto);
        all_errors.insert(all_errors.end(), errs2.begin(), errs2.end());
        auto errs3 = validate_graph_descriptors(proto);
        all_errors.insert(all_errors.end(), errs3.begin(), errs3.end());
        auto errs4 = validate_graph_topology_and_connections(proto);
        all_errors.insert(all_errors.end(), errs4.begin(), errs4.end());
        if (!all_errors.empty()) return all_errors;
    }

    {
        auto errs = validate_legacy_requirements(proto);
        all_errors.insert(all_errors.end(), errs.begin(), errs.end());
        if (!all_errors.empty()) return all_errors;
    }

    return all_errors;
}

void MeshGraphDescriptor::populate() {
    populate_descriptors();
    std::vector<NodeId> hierarchy;
    top_level_id_ = populate_instance(proto_->top_level_instance(), hierarchy);
}


std::vector<std::string> MeshGraphDescriptor::validate_basic_structure(const proto::MeshGraphDescriptor& proto) {
    std::vector<std::string> errors;
    if (proto.mesh_descriptors_size() == 0) {
        errors.push_back("There must be at least one mesh descriptor");
    }
    if (!proto.has_top_level_instance()) {
        errors.push_back("Top level instance is required");
    }
    return errors;
}



std::vector<std::string> MeshGraphDescriptor::validate_names(const proto::MeshGraphDescriptor& proto) {
    std::vector<std::string> error_messages;

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
    }

    return error_messages;
}


std::vector<std::string> MeshGraphDescriptor::validate_mesh_topology(const proto::MeshGraphDescriptor& proto) {
    std::vector<std::string> error_messages;

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
    }

    return error_messages;
}

std::vector<std::string> MeshGraphDescriptor::validate_architecture_consistency(const proto::MeshGraphDescriptor& proto) {
    std::vector<std::string> error_messages;

    // Check all architectures are the same
    if (proto.mesh_descriptors_size() > 0) {
        proto::Architecture first_arch = proto.mesh_descriptors(0).arch();
        if (!std::all_of(proto.mesh_descriptors().begin(), proto.mesh_descriptors().end(),
                        [first_arch](const auto& mesh) { return mesh.arch() == first_arch; })) {
            error_messages.push_back("All mesh descriptors must have the same architecture");
            return error_messages;
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
        uint32_t max_num_dims = get_max_dimensions_for_architecture(mesh.arch());
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
        
    return error_messages;
}

std::vector<std::string> MeshGraphDescriptor::validate_channels(const proto::MeshGraphDescriptor& proto) {
    std::vector<std::string> error_messages;

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

    // No graph_topology.channels() in current schema; skip

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

    return error_messages;
}

std::vector<std::string> MeshGraphDescriptor::validate_express_connections(const proto::MeshGraphDescriptor& proto) {
    std::vector<std::string> error_messages;

    // Validate express connections
    for (const auto& mesh : proto.mesh_descriptors()) {
        uint32_t num_dims = mesh.device_topology().dims_size();

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

    return error_messages;
}

std::vector<std::string> MeshGraphDescriptor::validate_graph_descriptors(const proto::MeshGraphDescriptor& proto) {
    std::vector<std::string> error_messages;

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

    return error_messages;
}

std::vector<std::string> MeshGraphDescriptor::validate_graph_topology_and_connections(const proto::MeshGraphDescriptor& proto) {
    std::vector<std::string> error_messages;

    // Combine all checks into a single loop over graph_descriptors
    for (const auto& graph : proto.graph_descriptors()) {
        // Check that there is a graph topology or connections for each graph descriptor
        if (!graph.has_graph_topology() && graph.connections_size() == 0) {
            error_messages.push_back( 
                fmt::format(
                    "Graph descriptor must have either graph_topology or connections defined (Graph: {})", 
                    graph.name()
                )
            );
            continue;
        }

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

    return error_messages;
}

std::vector<std::string> MeshGraphDescriptor::validate_legacy_requirements(const proto::MeshGraphDescriptor& proto) {
    std::vector<std::string> error_messages;

    // Validate that channels count must all be exactly the same
    uint32_t first_channels_count = proto.mesh_descriptors(0).channels().count();
    for (const auto& mesh : proto.mesh_descriptors()) {
        if (mesh.channels().count() != first_channels_count) {
            error_messages.push_back( 
                fmt::format( "Channel count must all be exactly the same (Mesh: {})", mesh.name()
            ));
        }
    }

    // Validate that device topology must all be exactly the same
    

    return error_messages;
}

void MeshGraphDescriptor::populate_descriptors() {
    mesh_desc_by_name_.clear();
    graph_desc_by_name_.clear();
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

MeshGraphDescriptor::NodeId MeshGraphDescriptor::populate_instance(const proto::NodeRef& node_ref, std::vector<NodeId>& hierarchy) {
    if (node_ref.has_mesh()) {
        return populate_mesh_instance(node_ref.mesh(), hierarchy);
    } else if (node_ref.has_graph()) {
        return populate_graph_instance(node_ref.graph(), hierarchy);
    }

    TT_THROW("Invalid NodeRef: neither mesh nor graph set");
    return -1;
}

MeshGraphDescriptor::NodeId MeshGraphDescriptor::populate_mesh_instance(const proto::MeshRef& mesh_ref, std::vector<NodeId>& hierarchy) {
    std::string_view descriptor_name = mesh_ref.mesh_descriptor();
    auto it = mesh_desc_by_name_.find(descriptor_name);
    TT_FATAL(it != mesh_desc_by_name_.end(), "Mesh descriptor {} not found in instance", std::string(descriptor_name));
    const auto* mesh_desc = it->second;

    InstanceData data{
        .local_id = mesh_ref.mesh_id(),
        .name = mesh_desc->name(),
        .type = "MESH",
        .kind = NodeKind::Mesh,
        .desc = mesh_desc,
    };

    auto emplace_result = instances_.emplace(data.global_id, std::move(data));
    auto & instance = emplace_result.first->second;

    hierarchy.push_back(instance.global_id);
    instance.hierarchy = hierarchy;
    hierarchy.pop_back();

    // Populate devices in the mesh
    uint32_t num_devices = 1;
    for (const auto& dim : mesh_desc->device_topology().dims()) {
        num_devices *= dim;
    }

    for (uint32_t i = 0; i < num_devices; ++i) {
        InstanceData device_data{
            .local_id = i,
            .name = "D" + std::to_string(i),
            .type = "DEVICE",
            .kind = NodeKind::Device,
            .desc = mesh_desc,
        };

        hierarchy.push_back(device_data.global_id);
        device_data.hierarchy = hierarchy;
        hierarchy.pop_back();

        auto emplace_result = instances_.emplace(device_data.global_id, std::move(device_data));
        auto & device = emplace_result.first->second;

        instance.sub_instances.insert(device.global_id);
        instance.sub_instances_local_id_to_global_id.emplace(i, device.global_id);
    }


    mesh_instances_.push_back(instance.global_id);
    instances_by_type_[std::string_view("MESH")].push_back(instance.global_id);
    instances_by_name_[mesh_desc->name()].push_back(instance.global_id);

    return instance.global_id;
}

MeshGraphDescriptor::NodeId MeshGraphDescriptor::populate_graph_instance(const proto::GraphRef& graph_ref, std::vector<NodeId>& hierarchy) {
    std::string_view descriptor_name = graph_ref.graph_descriptor();
    auto it = graph_desc_by_name_.find(descriptor_name);
    TT_FATAL(it != graph_desc_by_name_.end(), "Graph descriptor {} not found in instance", std::string(descriptor_name));
    const auto* graph_desc = it->second;

    InstanceData data{
        .local_id = graph_ref.graph_id(),
        .name = graph_desc->name(),
        .type = graph_desc->type(),
        .kind = NodeKind::Graph,
        .desc = graph_desc,
    };

    auto emplace_result = instances_.emplace(data.global_id, std::move(data));
    auto & instance = emplace_result.first->second;

    hierarchy.push_back(instance.global_id);
    instance.hierarchy = hierarchy;

    // Populate sub-instances from the graph descriptor
    std::unordered_set<NodeId> children_global_ids;
    children_global_ids.reserve(graph_desc->instances_size());

    std::string_view child_graph_type;
    for (const auto& sub_ref : graph_desc->instances()) {
        NodeId child = populate_instance(sub_ref, hierarchy);

        const auto & child_instance = instances_.at(child);

        // Check that the child instance created has the same type as rest of the graph descriptor
        if (child_instance.kind == NodeKind::Graph) {
            if (child_graph_type.empty()) {
                child_graph_type = child_instance.type;
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


void MeshGraphDescriptor::print_node(NodeId id, int indent_level) {
    std::string indent(indent_level * 2, ' ');
    std::stringstream ss;

    auto it = instances_.find(id);
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
            std::vector<std::pair<NodeId, NodeId>> ordered;
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
            std::vector<std::pair<NodeId, NodeId>> ordered;
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
        const auto* mesh_desc = std::get<const proto::MeshDescriptor*>(inst.desc);
        ss << indent << "Parent Mesh: " << mesh_desc->name() << std::endl;
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


// Connnection phase checks
// Check that all instances have been defined somewhere 

}  // namespace tt::tt_fabric

