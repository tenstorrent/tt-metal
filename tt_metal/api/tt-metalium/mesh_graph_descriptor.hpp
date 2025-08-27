// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>
#include <memory>
#include <vector>

namespace tt::tt_fabric {

namespace proto {
class MeshGraphDescriptor;
}

class MeshGraphDescriptor {

class GraphInstance {
    std::string name;
    std::string type;
    std::vector<NodeInstance*> node_instances;
};

class MeshInstance {
    std::string name;
    std::array<int, 2> topology;
};


public:
    explicit MeshGraphDescriptor(const std::string& text_proto);
    explicit MeshGraphDescriptor(const std::filesystem::path& text_proto_file_path);
    ~MeshGraphDescriptor();

private:
    std::unique_ptr<proto::MeshGraphDescriptor> proto_;
    static bool static_validate(const proto::MeshGraphDescriptor& proto);
    static void set_defaults(proto::MeshGraphDescriptor& proto);

    void instantiate_mesh_graph();
};

}  // namespace tt::tt_fabric
