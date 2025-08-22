// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
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

}  // namespace

MeshGraphDescriptor::MeshGraphDescriptor(const std::string &text_proto, bool allow_unknown_fields, bool allow_unknown_extensions) {
    tt::tt_fabric::mesh_graph_descriptor_proto::MeshGraphDescriptor temp_proto;
    google::protobuf::TextFormat::Parser parser;
    parser.AllowUnknownField(allow_unknown_fields);
    parser.AllowUnknownExtension(allow_unknown_extensions);
    bool result = parser.ParseFromString(text_proto, &temp_proto);
    TT_FATAL(result, "Failed to parse MeshGraphDescriptor textproto");

    // Validate the proto
    result = static_validate(temp_proto);
    TT_FATAL(result, "Failed to validate MeshGraphDescriptor textproto");
}

MeshGraphDescriptor::MeshGraphDescriptor(const std::filesystem::path &text_proto_file_path, bool allow_unknown_fields, bool allow_unknown_extensions) 
    : MeshGraphDescriptor(read_file_to_string(text_proto_file_path.string()), allow_unknown_fields, allow_unknown_extensions) {
}

bool MeshGraphDescriptor::static_validate(const mesh_graph_descriptor_proto::MeshGraphDescriptor& proto) {
    bool success = true;

    // There has to exist at least one mesh or graph descriptor
    if (proto.mesh_descriptors_size() == 0 && proto.graph_descriptors_size() == 0) {
        log_error(LogFabric, "MeshGraphDescriptor: No mesh or graph descriptors defined");
        success = false;
    }

    // Validate that all mesh descriptors are valid
    for (const auto& mesh : proto.mesh_descriptors()) {
        // Check that all dimenstions are the same size
        uint32_t golden_num_dims = ARCH_TO_NUM_DIMS.at(mesh.arch());
        uint32_t num_dims = std::min(static_cast<uint32_t>(mesh.device_topology().dims_size()), golden_num_dims);

        if (mesh.device_topology().types_size() > 0) {
            if (mesh.device_topology().dims_size() != mesh.device_topology().types_size()) {
                log_error(LogFabric, "MeshGraphDescriptor: Device topology dimensions and types must be the same size");
                success = false;
            }
        }

        if (mesh.device_topology().dims_size() != mesh.host_topology().dims_size()) {
            log_error(LogFabric, "MeshGraphDescriptor: Device and host topology dimensions must be the same size");
            success = false;
        }

        if (mesh.device_topology().dims_size() != golden_num_dims) {
            log_error(LogFabric, "MeshGraphDescriptor: {} architecture devices require {} dimensions, but {} were provided", mesh.arch(), golden_num_dims, mesh.device_topology().dims_size());
            success = false;
        }

        // Calculate the number of devices in the mesh
        uint32_t num_devices = 1;
        for (uint32_t i = 0; i < num_dims; i++) {
            num_devices *= mesh.device_topology().dims(i);
        }

        // Check that express connections are valid
        for (const auto& express_connection : mesh.express_connections()) {
            if (express_connection.src() < 0 || express_connection.src() >= num_devices) {
                log_error(LogFabric, "MeshGraphDescriptor: Express connection source is out of bounds");
                success = false;
            }
            if (express_connection.dst() < 0 || express_connection.dst() >= num_devices) {
                log_error(LogFabric, "MeshGraphDescriptor: Express connection destination is out of bounds");
                success = false;
            }
        }

    }


    return success;
}

}  // namespace tt::tt_fabric


