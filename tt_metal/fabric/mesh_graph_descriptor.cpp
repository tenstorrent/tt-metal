// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "tt-metalium/mesh_graph_descriptor.hpp"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

// Generated from mesh_graph_descriptor.proto
#include "mesh_graph_descriptor_schema/mesh_graph_descriptor.pb.h"

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

MeshGraphDescriptor::MeshGraphDescriptor(const std::string &text_proto) {
    tt::tt_fabric::mesh_graph_descriptor_proto::MeshGraphDescriptor proto;
    if (!google::protobuf::TextFormat::ParseFromString(text_proto, &proto)) {
        throw std::runtime_error("Failed to parse MeshGraphDescriptor textproto");
    }
    // Intentionally do not expose or store parsed state in the header. This validates and
    // constructs an internal representation only when protobuf is enabled.
}

MeshGraphDescriptor::MeshGraphDescriptor(const std::filesystem::path &text_proto_file_path) {
    const std::string content = read_file_to_string(text_proto_file_path.string());
    tt::tt_fabric::mesh_graph_descriptor_proto::MeshGraphDescriptor proto;
    if (!google::protobuf::TextFormat::ParseFromString(content, &proto)) {
        throw std::runtime_error("Failed to parse MeshGraphDescriptor textproto file: " + text_proto_file_path.string());
    }
}

}  // namespace tt::tt_fabric


