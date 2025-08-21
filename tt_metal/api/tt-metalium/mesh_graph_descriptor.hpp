// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>

namespace tt::tt_fabric {

// Minimal, header-only interface for MeshGraphDescriptor.
// Implementation that depends on Protobuf lives in the corresponding .cpp
// to keep imports minimal for dependents.
class MeshGraphDescriptor {
public:
    MeshGraphDescriptor(const std::string &text_proto);
    MeshGraphDescriptor(const std::filesystem::path &text_proto_file_path);

private:
};

}  // namespace tt::tt_fabric
