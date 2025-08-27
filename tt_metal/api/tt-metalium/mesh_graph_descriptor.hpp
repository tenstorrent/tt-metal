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
    explicit MeshGraphDescriptor(const std::string& text_proto);
    explicit MeshGraphDescriptor(const std::filesystem::path& text_proto_file_path);
    ~MeshGraphDescriptor();

private:
    std::unique_ptr<proto::MeshGraphDescriptor> proto_;
    static bool static_validate(const proto::MeshGraphDescriptor& proto);
};

}  // namespace tt::tt_fabric
