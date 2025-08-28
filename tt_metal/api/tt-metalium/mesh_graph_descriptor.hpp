// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <utility>

namespace tt::tt_fabric {

namespace proto {
class MeshGraphDescriptor;
class MeshDescriptor;
class GraphDescriptor;
class NodeRef;
class MeshRef;
class GraphRef;
}

class MeshGraphDescriptor {

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
    explicit MeshGraphDescriptor(const std::string& text_proto);
    explicit MeshGraphDescriptor(const std::filesystem::path& text_proto_file_path);
    ~MeshGraphDescriptor();

private:
    std::unique_ptr<proto::MeshGraphDescriptor> proto_;

    std::unordered_map<std::string, const proto::MeshDescriptor*> mesh_descriptors_;
    std::unordered_map<std::string, const proto::GraphDescriptor*> graph_descriptors_;

    std::unordered_map<uint32_t, std::unique_ptr<NodeInstance>> node_instances_;
    NodeInstance* top_level_instance_;

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
    static std::vector<std::string> validate_legacy_requirements(const proto::MeshGraphDescriptor& proto);

    void save_descriptor_refs();

    void construct_graph_instances();
    std::unique_ptr<NodeInstance> construct_graph_instance(const proto::NodeRef& node_ref);

};

}  // namespace tt::tt_fabric
