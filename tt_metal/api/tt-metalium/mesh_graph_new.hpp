// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <atomic>

namespace tt::tt_fabric {

namespace proto {
class MeshGraphDescriptor;
class MeshDescriptor;
class GraphDescriptor;
class NodeRef;
class MeshRef;
class GraphRef;
}

class MeshGraphNew {

struct NodeInstance {
    std::string descriptor_name;
    uint32_t id;
    uint32_t global_id = generate_next_global_id();

    NodeInstance() = default;
    NodeInstance(const std::string& descriptor_name, uint32_t id)
        : descriptor_name(descriptor_name), id(id) {}

private:
    inline static uint32_t generate_next_global_id() {
        static std::atomic_uint32_t next_global_id_ = 0;
        return next_global_id_++;
    }
};

struct MeshInstance : public NodeInstance {
    std::vector<uint32_t> dimensions;
    std::unordered_set<uint32_t> device_ids;
    MeshInstance() = default;
    MeshInstance(const std::string& descriptor_name, uint32_t id, std::vector<uint32_t> dimensions)
        : NodeInstance(descriptor_name, id), dimensions(dimensions), device_ids(generate_device_ids(dimensions)) {}

private:
    inline static std::unordered_set<uint32_t> generate_device_ids(const std::vector<uint32_t>& dimensions) {
        std::unordered_set<uint32_t> device_ids;
        uint32_t num_devices = 1;
        for (const auto& dim : dimensions) {
            num_devices *= dim;
        }
        for (uint32_t i = 0; i < num_devices; i++) {
            device_ids.insert(i);
        }
        return device_ids;
    }
};

struct GraphInstance : public NodeInstance {
    std::unordered_set<uint32_t> node_ids = {};
    GraphInstance() = delete;
    GraphInstance(const std::string& descriptor_name, uint32_t id)
        : NodeInstance(descriptor_name, id) {}
};

public:
    explicit MeshGraphNew(const proto::MeshGraphDescriptor& proto);
    ~MeshGraphNew() = default;

    // Getters for accessing the instances
    const std::unordered_map<uint32_t, std::unique_ptr<NodeInstance>>& get_node_instances() const { return node_instances_; }
    const NodeInstance* get_top_level_instance() const { return top_level_instance_; }

private:

    std::unordered_map<uint32_t, std::unique_ptr<NodeInstance>> node_instances_;
    NodeInstance* top_level_instance_;

    void save_descriptor_refs(const proto::MeshGraphDescriptor& proto);
    void construct_graph_instances(const proto::MeshGraphDescriptor& proto);
    std::unique_ptr<NodeInstance> construct_graph_instance(const proto::NodeRef& node_ref);
};

}  // namespace tt::tt_fabric
