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

uint32_t get_max_dimensions_for_architecture(proto::Architecture arch) {
    switch (arch) {
        case proto::Architecture::WORMHOLE_B0: return 2;
        case proto::Architecture::BLACKHOLE: return 3;
        default: TT_FATAL(false, "Invalid architecture"); return 0;
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

    // Validate the proto
    TT_FATAL(static_validate(temp_proto), "Failed to validate MeshGraphDescriptor textproto");

    proto_ = std::make_unique<proto::MeshGraphDescriptor>(temp_proto);
}

MeshGraphDescriptor::MeshGraphDescriptor(const std::filesystem::path& text_proto_file_path) :
    MeshGraphDescriptor(read_file_to_string(text_proto_file_path.string())) {}

MeshGraphDescriptor::~MeshGraphDescriptor() = default;

bool MeshGraphDescriptor::static_validate(const proto::MeshGraphDescriptor& proto) {
    bool success = true;

    // There has to exist at least one mesh or graph descriptor
    if (proto.mesh_descriptors_size() == 0 && proto.graph_descriptors_size() == 0) {
        log_error(LogFabric, "MeshGraphDescriptor: No mesh or graph descriptors defined");
        success = false;
    }

    // Validate that all mesh descriptors are valid
    for (const auto& mesh : proto.mesh_descriptors()) {
        // Check that all dimenstions are the same size
        uint32_t max_num_dims = get_max_dimensions_for_architecture(mesh.arch());
        uint32_t num_dims = std::min(static_cast<uint32_t>(mesh.device_topology().dims_size()), max_num_dims);

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

        if (mesh.device_topology().dims_size() > max_num_dims) {
            log_error(
                LogFabric,
                "MeshGraphDescriptor: {} architecture devices allow a maximum of {} dimensions, but {} were provided",
                mesh.arch(),
                max_num_dims,
                mesh.device_topology().dims_size());
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
