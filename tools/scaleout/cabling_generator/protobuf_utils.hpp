// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>
#include <google/protobuf/text_format.h>
#include <stdexcept>
#include <string>

#include "protobuf/cluster_config.pb.h"

namespace tt::scaleout_tools {

template <typename Descriptor>
[[nodiscard]] Descriptor load_descriptor_from_textproto(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    const std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    Descriptor descriptor;
    if (!google::protobuf::TextFormat::ParseFromString(file_content, &descriptor)) {
        throw std::runtime_error("Failed to parse textproto file: " + file_path);
    }
    return descriptor;
}

[[nodiscard]] inline cabling_generator::proto::ClusterDescriptor load_cluster_descriptor(const std::string& file_path) {
    return load_descriptor_from_textproto<cabling_generator::proto::ClusterDescriptor>(file_path);
}

}  // namespace tt::scaleout_tools
