// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <set>
#include "assert.hpp"

#include "tt-metalium/mesh_graph_descriptor.hpp"
#include "protobuf/mesh_graph_descriptor.pb.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

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
        default: log_error(LogFabric, "Invalid architecture"); return 0;
    }
}

}  // namespace

MeshGraphDescriptor::MeshGraphDescriptor(const std::string& text_proto) {
    proto::MeshGraphDescriptor temp_proto;
    google::protobuf::TextFormat::Parser parser;

    // Allowing for back and forward compatibility for fields not currently in the proto file
    parser.AllowUnknownField(true);
    parser.AllowUnknownExtension(true);

    TT_FATAL(parser.ParseFromString(text_proto, &temp_proto), "Failed to parse MeshGraphDescriptor textproto");

    // Set defaults for missing fields
    set_defaults(temp_proto);

    // Validate the proto
    TT_FATAL(static_validate(temp_proto), "Failed to validate MeshGraphDescriptor textproto");

    proto_ = std::make_unique<proto::MeshGraphDescriptor>(temp_proto);
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

bool MeshGraphDescriptor::static_validate(const proto::MeshGraphDescriptor& proto) {
    bool success = true;

    // ============================================================================
    // BASIC STRUCTURE VALIDATION
    // ============================================================================
    
    // There has to exist at least one mesh or graph descriptor
    if (proto.mesh_descriptors_size() == 0) {
        log_error(LogFabric, "MeshGraphDescriptor: There must be at least one mesh descriptor");
        success = false;
    }

    // ============================================================================
    // MESH DESCRIPTOR VALIDATION
    // ============================================================================
    
    // Validate basic mesh properties (names and dimensions)
    for (const auto& mesh : proto.mesh_descriptors()) {
        // Check that the name is not empty
        if (mesh.name().empty()) {
            log_error(LogFabric, "MeshGraphDescriptor: Mesh descriptor name cannot be empty");
            success = false;
        }

        // Check that all dims are positive
        for (const auto& dim : mesh.device_topology().dims()) {
            if (dim <= 0) {
                log_error(LogFabric, "MeshGraphDescriptor: Device topology dimensions must be positive");
                success = false;
            }
        }

        // Check that device topology dimensions and types are the same size
        if (mesh.device_topology().dim_types_size() > 0) {
            if (mesh.device_topology().dims_size() != mesh.device_topology().dim_types_size()) {
                log_error(LogFabric, "MeshGraphDescriptor: Device topology dimensions and types must be the same size");
                success = false;
            }
        }
    }

    // ============================================================================
    // TOPOLOGY VALIDATION
    // ============================================================================
    
    // Validate topology consistency between device and host
    for (const auto& mesh : proto.mesh_descriptors()) {
        // Check that the device and host topology dimensions are the same size
        if (mesh.device_topology().dims_size() != mesh.host_topology().dims_size()) {
            log_error(LogFabric, "MeshGraphDescriptor: Device and host topology dimensions must be the same size");
            success = false;
        }
    }

    // Validate architecture and dimension limits
    for (const auto& mesh : proto.mesh_descriptors()) {
        uint32_t max_num_dims = get_max_dimensions_for_architecture(mesh.arch());
        if (max_num_dims == 0) {
            log_error(LogFabric, "MeshGraphDescriptor: Invalid architecture");
            success = false;
        }
        
        // Check that the number of dimensions is not greater than the maximum allowed for the architecture
        if (mesh.device_topology().dims_size() > max_num_dims) {
            log_error(
                LogFabric,
                "MeshGraphDescriptor: {} architecture devices allow a maximum of {} dimensions, but {} were provided",
                mesh.arch(),
                max_num_dims,
                mesh.device_topology().dims_size());
            success = false;
        }
    }

    // Set dim_types to LINE if not specified for each dimension
    // Note: This is handled during parsing/processing, not validation
    // The validation here is to ensure dim_types are valid when specified
    for (const auto& mesh : proto.mesh_descriptors()) {
        for (const auto& dim_type : mesh.device_topology().dim_types()) {
            if (dim_type == proto::TorusTopology::INVALID_TYPE) {
                log_error(LogFabric, "MeshGraphDescriptor: Mesh descriptor '{}' has invalid dimension type", mesh.name());
                success = false;
            }
        }
    }

    // ============================================================================
    // REQUIRED FIELDS VALIDATION
    // ============================================================================
    
    // Verify that arch, device and host topology must exist in mesh descriptors
    for (const auto& mesh : proto.mesh_descriptors()) {
        if (mesh.arch() == proto::Architecture::INVALID_ARCHITECTURE) {
            log_error(LogFabric, "MeshGraphDescriptor: Mesh descriptor '{}' must have a valid architecture", mesh.name());
            success = false;
        }
        
        if (!mesh.has_device_topology() || mesh.device_topology().dims_size() == 0) {
            log_error(LogFabric, "MeshGraphDescriptor: Mesh descriptor '{}' must have device topology with dimensions", mesh.name());
            success = false;
        }
        
        if (!mesh.has_host_topology() || mesh.host_topology().dims_size() == 0) {
            log_error(LogFabric, "MeshGraphDescriptor: Mesh descriptor '{}' must have host topology with dimensions", mesh.name());
            success = false;
        }
    }

    // ============================================================================
    // ARCHITECTURE CONSISTENCY VALIDATION
    // ============================================================================
    
    // Check all architectures are the same
    if (proto.mesh_descriptors_size() > 0) {
        proto::Architecture first_arch = proto.mesh_descriptors(0).arch();
        for (const auto& mesh : proto.mesh_descriptors()) {
            if (mesh.arch() != first_arch) {
                log_error(LogFabric, "MeshGraphDescriptor: All mesh descriptors must have the same architecture");
                success = false;
            }
        }
    }

    // Check that all mesh descriptors have the same architecture (if they reference meshes)
    // NOTE: In the future we might allow for different architectures in the same graph
    std::set<proto::Architecture> architectures;
    for (const auto& mesh : proto.mesh_descriptors()) {
        architectures.insert(mesh.arch());
    }
    if (architectures.size() > 1) {
        log_error(LogFabric, "MeshGraphDescriptor: All mesh descriptors must have the same architecture");
        success = false;
    }

    // ============================================================================
    // UNIQUENESS VALIDATION
    // ============================================================================
    
    // Check that all mesh descriptors have a unique name
    std::set<std::string> mesh_names;
    for (const auto& mesh : proto.mesh_descriptors()) {
        if (mesh.name().empty()) {
            log_error(LogFabric, "MeshGraphDescriptor: Mesh descriptor name cannot be empty");
            success = false;
        }
        if (mesh_names.find(mesh.name()) != mesh_names.end()) {
            log_error(LogFabric, "MeshGraphDescriptor: Mesh descriptor name '{}' is not unique", mesh.name());
            success = false;
        }
        mesh_names.insert(mesh.name());
    }

    // Check that all graph descriptors have a unique name
    std::set<std::string> names;
    for (const auto& graph : proto.graph_descriptors()) {
        if (graph.name().empty()) {
            log_error(LogFabric, "MeshGraphDescriptor: Graph descriptor name cannot be empty");
            success = false;
        }
        if (names.find(graph.name()) != names.end()) {
            log_error(LogFabric, "MeshGraphDescriptor: Graph descriptor name '{}' is not unique", graph.name());
            success = false;
        }
        names.insert(graph.name());
    }
    for (const auto& mesh : proto.mesh_descriptors()) {
        if (names.find(mesh.name()) != names.end()) {
            log_error(LogFabric, "MeshGraphDescriptor: Mesh descriptor name '{}' is not unique", mesh.name());
            success = false;
        }
        names.insert(mesh.name());
    }

    // ============================================================================
    // CHANNEL VALIDATION
    // ============================================================================
    
    // Check all channel counts > 0
    for (const auto& mesh : proto.mesh_descriptors()) {
        if (mesh.channels().count() <= 0) {
            log_error(LogFabric, "MeshGraphDescriptor: Channel count must be positive");
            success = false;
        }
    }

    // Check that channels in graph topology are positive
    for (const auto& graph : proto.graph_descriptors()) {
        if (graph.has_graph_topology() && graph.graph_topology().channels().count() <= 0) {
            log_error(LogFabric, "MeshGraphDescriptor: Graph Descriptor '{}' channel count must be positive", graph.name());
            success = false;
        }
    }

    // Check all channel counts > 0 in graph descriptors and connections
    for (const auto& graph : proto.graph_descriptors()) {
        // Check connection-level channels and validate connection nodes
        for (const auto& connection : graph.connections()) {
            if (connection.channels().count() <= 0) {
                log_error(LogFabric, "MeshGraphDescriptor: Connection in graph '{}' channel count must be positive", graph.name());
                success = false;
            }
            // Check there is at least one node in the connection
            if (connection.nodes_size() < 2) {
                log_error(LogFabric, "MeshGraphDescriptor: Connection in graph '{}' must have at least two nodes", graph.name());
                success = false;
            }
        }
    }

    // ============================================================================
    // POLICY VALIDATION
    // ============================================================================
    
    // Set the default for channel policy to strict if not specified
    // Note: This is handled during parsing/processing, not validation
    // The validation here is to ensure channels have valid policies when specified
    for (const auto& mesh : proto.mesh_descriptors()) {
        if (mesh.has_channels() && mesh.channels().has_policy()) {
            if (mesh.channels().policy() == proto::Policy::INVALID_POLICY) {
                log_error(LogFabric, "MeshGraphDescriptor: Mesh descriptor '{}' has invalid channel policy", mesh.name());
                success = false;
            }
        }
    }
    
    for (const auto& graph : proto.graph_descriptors()) {
        // Check graph topology channels policy
        if (graph.has_graph_topology() && graph.graph_topology().has_channels() && 
            graph.graph_topology().channels().has_policy()) {
            if (graph.graph_topology().channels().policy() == proto::Policy::INVALID_POLICY) {
                log_error(LogFabric, "MeshGraphDescriptor: Graph descriptor '{}' graph topology has invalid channel policy", graph.name());
                success = false;
            }
        }
        
        // Check connection channels policy
        for (const auto& connection : graph.connections()) {
            if (connection.has_channels() && connection.channels().has_policy()) {
                if (connection.channels().policy() == proto::Policy::INVALID_POLICY) {
                    log_error(LogFabric, "MeshGraphDescriptor: Connection in graph '{}' has invalid channel policy", graph.name());
                    success = false;
                }
            }
        }
    }

    // ============================================================================
    // EXPRESS CONNECTIONS VALIDATION
    // ============================================================================
    
    // Validate express connections
    for (const auto& mesh : proto.mesh_descriptors()) {
        uint32_t max_num_dims = get_max_dimensions_for_architecture(mesh.arch());
        if (max_num_dims == 0) {
            log_error(LogFabric, "MeshGraphDescriptor: Invalid architecture");
            success = false;
        }
        uint32_t num_dims = std::min(static_cast<uint32_t>(mesh.device_topology().dims_size()), max_num_dims);

        // Calculate the number of devices in the mesh
        uint32_t num_devices = 1;
        for (uint32_t i = 0; i < num_dims; i++) {
            num_devices *= mesh.device_topology().dims(i);
        }

        // Check that express connections are valid and have the right number of devices
        for (const auto& express_connection : mesh.express_connections()) {
            if (express_connection.src() < 0 || express_connection.src() >= num_devices) {
                log_error(LogFabric, "MeshGraphDescriptor: Express connection source is out of bounds for mesh '{}'", mesh.name());
                success = false;
            }
            if (express_connection.dst() < 0 || express_connection.dst() >= num_devices) {
                log_error(LogFabric, "MeshGraphDescriptor: Express connection destination is out of bounds for mesh '{}'", mesh.name());
                success = false;
            }
        }
    }

    // ============================================================================
    // GRAPH DESCRIPTOR VALIDATION
    // ============================================================================
    
    // Check that there is at least one instance in the graph and validate references
    for (const auto& graph : proto.graph_descriptors()) {
        if (graph.instances_size() == 0) {
            log_error(LogFabric, "MeshGraphDescriptor: Graph descriptor '{}' must have at least one instance", graph.name());
            success = false;
        }
    }

    // Verify that type is set in graph descriptors
    for (const auto& graph : proto.graph_descriptors()) {
        if (graph.type().empty()) {
            log_error(LogFabric, "MeshGraphDescriptor: Graph descriptor '{}' must have a type specified", graph.name());
            success = false;
        }
    }

    // ============================================================================
    // GRAPH TOPOLOGY AND CONNECTIONS VALIDATION
    // ============================================================================
    
    // Check that there is a graph topology or connections for each graph descriptor       
    for (const auto& graph : proto.graph_descriptors()) {
        if (!graph.has_graph_topology() && graph.connections_size() == 0) {
            log_error(LogFabric, "MeshGraphDescriptor: Graph descriptor '{}' must have either graph_topology or connections defined", graph.name());
            success = false;
        }
    }

    // Check that when graph_topology is used, connections aren't defined
    for (const auto& graph : proto.graph_descriptors()) {
        if (graph.has_graph_topology() && graph.connections_size() > 0) {
            log_error(LogFabric, "MeshGraphDescriptor: Graph descriptor '{}' cannot have both graph_topology and connections defined", graph.name());
            success = false;
        }
    }

    // Check that when connections are used, graph_topology isn't defined
    for (const auto& graph : proto.graph_descriptors()) {
        if (graph.connections_size() > 0 && graph.has_graph_topology()) {
            log_error(LogFabric, "MeshGraphDescriptor: Graph descriptor '{}' cannot have both connections and graph_topology defined", graph.name());
            success = false;
        }
    }

    return success;
}

// Dynamic checks to implement in seperate PR
// Check that all instances have been defined somewhere
// Check that all instances are of the same type
// Validate that connection nodes reference valid instances in this graph

}  // namespace tt::tt_fabric
