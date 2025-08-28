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

    // Error reporting methods
    std::string get_validation_report() const;
    const std::vector<std::string>& get_error_messages() const { return error_messages_; }

private:
    std::unique_ptr<proto::MeshGraphDescriptor> proto_;
    std::vector<std::string> error_messages_;

    static void set_defaults(proto::MeshGraphDescriptor& proto);
    bool static_validate(const proto::MeshGraphDescriptor& proto);
    
    // Helper methods for validation with error collection
    bool validate_basic_structure(const proto::MeshGraphDescriptor& proto);
    bool validate_names(const proto::MeshGraphDescriptor& proto);
    bool validate_mesh_topology(const proto::MeshGraphDescriptor& proto);
    bool validate_architecture_consistency(const proto::MeshGraphDescriptor& proto);
    bool validate_channels(const proto::MeshGraphDescriptor& proto);
    bool validate_express_connections(const proto::MeshGraphDescriptor& proto);
    bool validate_graph_descriptors(const proto::MeshGraphDescriptor& proto);
    bool validate_graph_topology_and_connections(const proto::MeshGraphDescriptor& proto);
};

}  // namespace tt::tt_fabric
