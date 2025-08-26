// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>
#include <memory>

namespace tt::tt_fabric {

namespace proto {
class MeshGraphDescriptor;
}

class MeshGraphDescriptor {
public:
    MeshGraphDescriptor(const std::string& text_proto);
    MeshGraphDescriptor(const std::filesystem::path& text_proto_file_path);
    ~MeshGraphDescriptor();

private:
    std::unique_ptr<proto::MeshGraphDescriptor> proto_;
};

}  // namespace tt::tt_fabric
