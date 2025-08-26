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
    proto::MeshGraphDescriptor temp_proto;
    google::protobuf::TextFormat::Parser parser;
    parser.AllowUnknownField(allow_unknown_fields);
    parser.AllowUnknownExtension(allow_unknown_extensions);
    bool result = parser.ParseFromString(text_proto, &temp_proto);
    TT_FATAL(result, "Failed to parse MeshGraphDescriptor textproto");

    // TODO: Add validation here

    proto_ = std::make_unique<proto::MeshGraphDescriptor>(temp_proto);
}

MeshGraphDescriptor::MeshGraphDescriptor(
    const std::filesystem::path& text_proto_file_path, bool allow_unknown_fields, bool allow_unknown_extensions) :
    MeshGraphDescriptor(
        read_file_to_string(text_proto_file_path.string()), allow_unknown_fields, allow_unknown_extensions) {}

MeshGraphDescriptor::~MeshGraphDescriptor() {
    // TODO Implement this
}

}  // namespace tt::tt_fabric
