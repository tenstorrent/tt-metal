// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>
#include <memory>
#include <vector>
#include <unordered_map>

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

    static void set_defaults(proto::MeshGraphDescriptor& proto);
    static std::vector<std::string> static_validate(const proto::MeshGraphDescriptor& proto);
    
    // Helper methods for validation that return their own error lists
    static std::vector<std::string> validate_basic_structure(const proto::MeshGraphDescriptor& proto);
    static std::vector<std::string> validate_names(const proto::MeshGraphDescriptor& proto);
    static std::vector<std::string> validate_mesh_topology(const proto::MeshGraphDescriptor& proto);
    static std::vector<std::string> validate_architecture_consistency(const proto::MeshGraphDescriptor& proto);
    static std::vector<std::string> validate_channels(const proto::MeshGraphDescriptor& proto);
    static std::vector<std::string> validate_express_connections(const proto::MeshGraphDescriptor& proto);
    static std::vector<std::string> validate_graph_descriptors(const proto::MeshGraphDescriptor& proto);
    static std::vector<std::string> validate_graph_topology_and_connections(const proto::MeshGraphDescriptor& proto);
};

}  // namespace tt::tt_fabric
