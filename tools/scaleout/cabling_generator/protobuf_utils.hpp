// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>
#include <google/protobuf/text_format.h>
#include <stdexcept>
#include <string>

namespace tt::scaleout_tools {

// Helper to load protobuf descriptors from textproto files
template <typename Descriptor>
Descriptor load_descriptor_from_textproto(const std::string& file_path) {
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

}  // namespace tt::scaleout_tools
