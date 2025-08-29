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
#include <variant>

// Forward declaration
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
    const uint32_t id;
    const std::string descriptor_name;
    std::variant<const proto::MeshDescriptor*, const proto::GraphDescriptor*> descriptor;
    std::variant<const proto::MeshRef*, const proto::GraphRef*> ref;
};

struct MeshInstance : public NodeInstance {
    const std::vector<uint32_t> dimensions;
};

struct GraphInstance : public NodeInstance {
    std::string type;
};



public:
    explicit MeshGraphDescriptor(const std::string& text_proto);
    explicit MeshGraphDescriptor(const std::filesystem::path& text_proto_file_path);
    ~MeshGraphDescriptor();

private:
    std::unique_ptr<const proto::MeshGraphDescriptor> proto_;


    std::unordered_map<std::string, const proto::MeshDescriptor*> mesh_descriptors_by_name_;
    std::unordered_map<std::string, const proto::GraphDescriptor*> graph_descriptors_by_name_;
    std::unordered_map<std::string, std::unordered_set<const proto::GraphDescriptor*>> graph_descriptors_by_type_;

    // TOOD: Organize the Instances by type and hash by name
    std::unordered_set<std::shared_ptr<NodeInstance>> all_instances_;
    std::shared_ptr<NodeInstance> top_level_instance_;


    // Static methods for setting defaults and validation
    static void set_defaults(proto::MeshGraphDescriptor& proto);

    // Static validation methods
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

    // Populate Descriptors and Instances
    void populate();
    // helper to populate descriptors
    void populate_descriptors();

    void populate_instances(const proto::NodeRef& node_ref);

    std::shared_ptr<NodeInstance> construct_node_instance(const proto::NodeRef& node_ref);
    

};

}  // namespace tt::tt_fabric
