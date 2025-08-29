// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "assert.hpp"
#include "protobuf/mesh_graph_descriptor.pb.h"
#include "tt-metalium/mesh_graph_new.hpp"

namespace tt::tt_fabric {

MeshGraphNew::MeshGraphNew(const proto::MeshGraphDescriptor& proto) {
    save_descriptor_refs(proto);
    construct_graph_instances(proto);
}

void MeshGraphNew::construct_graph_instances(const proto::MeshGraphDescriptor& proto) {
    auto top_inst = construct_graph_instance(proto.top_level_instance());
    auto top_gid = top_inst->global_id;

    node_instances_[top_gid] = std::move(top_inst);

    top_level_instance_ = node_instances_[top_gid].get();
}

std::unique_ptr<MeshGraphNew::NodeInstance> MeshGraphNew::construct_graph_instance(const proto::NodeRef& node_ref) {
    if (node_ref.has_mesh()) {
        // Check the the mesh descriptor exists
        auto it = mesh_descriptors_.find(node_ref.mesh().mesh_descriptor());
        TT_FATAL(it != mesh_descriptors_.end(),
            "Mesh descriptor {} not found",
            node_ref.mesh().mesh_descriptor());

        const auto& mesh_descriptor = it->second;

        std::vector<uint32_t> dimensions;
        for (const auto& dim : mesh_descriptor->device_topology().dims()) {
            dimensions.push_back(dim);
        }

        MeshInstance mesh_instance(
            node_ref.mesh().mesh_descriptor(),
            node_ref.mesh().mesh_id(),
            dimensions
        );

        return std::make_unique<MeshInstance>(mesh_instance);

    } else if (node_ref.has_graph()) {

        auto it = graph_descriptors_.find(node_ref.graph().graph_descriptor());

        TT_FATAL(it != graph_descriptors_.end(),
            "Graph descriptor {} not found", 
            node_ref.graph().graph_descriptor());

        const auto& graph_descriptor = it->second;

        GraphInstance graph_instance(
            node_ref.graph().graph_descriptor(),
            node_ref.graph().graph_id()
        );

        for (const auto& instance : graph_descriptor->instances()) {
            auto sub_instance = construct_graph_instance(instance);
            auto sub_gid = sub_instance->global_id;

            node_instances_[sub_gid] = std::move(sub_instance);

            auto [it, inserted] = graph_instance.node_ids.insert(sub_gid);
            TT_FATAL(inserted, "Node ID {} already exists in graph instance", sub_gid);
        }

        return std::make_unique<GraphInstance>(graph_instance);
    }

    return {};
}

}  // namespace tt::tt_fabric
