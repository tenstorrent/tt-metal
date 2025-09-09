// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <set>
#include <unordered_set>
#include "assert.hpp"

#include "protobuf/mesh_graph_descriptor.pb.h"
#include "tt-metalium/mesh_graph_descriptor.hpp"

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
    auto errors = static_validate(temp_proto);
    TT_FATAL(errors.empty(), "Failed to validate MeshGraphDescriptor textproto: \n{}", get_validation_report(errors));

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

std::vector<std::string> MeshGraphDescriptor::static_validate(const proto::MeshGraphDescriptor& proto) {
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

    return all_errors;
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

}

// Dynamic checks to implement
// Check that all instances have been defined somewhere
// Check that all instances are of the same type
// Validate that connection nodes reference valid instances in this graph

}  // namespace tt::tt_fabric
