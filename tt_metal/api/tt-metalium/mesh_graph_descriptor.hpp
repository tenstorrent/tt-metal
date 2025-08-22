// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>

#include "protobuf/mesh_graph_descriptor.pb.h"

namespace tt::tt_fabric {

const std::unordered_map<mesh_graph_descriptor_proto::Architecture, uint32_t> ARCH_TO_NUM_DIMS = {
    {mesh_graph_descriptor_proto::Architecture::WORMHOLE_B0, 2},
    {mesh_graph_descriptor_proto::Architecture::BLACKHOLE, 3},
};

class MeshGraphDescriptor {
public:
    MeshGraphDescriptor(const std::string &text_proto, bool allow_unknown_fields = true, bool allow_unknown_extensions = false);
    MeshGraphDescriptor(const std::filesystem::path &text_proto_file_path, bool allow_unknown_fields = true, bool allow_unknown_extensions = false);

private:
    std::unique_ptr<mesh_graph_descriptor_proto::MeshGraphDescriptor> proto_;

    static bool static_validate(const mesh_graph_descriptor_proto::MeshGraphDescriptor& proto);

    // TODO: Implement this
    bool populate_mesh_graph(const std::string& mesh_graph_desc_file);
    void validate_mesh_graph_against_proto();
    
};

}  // namespace tt::tt_fabric
