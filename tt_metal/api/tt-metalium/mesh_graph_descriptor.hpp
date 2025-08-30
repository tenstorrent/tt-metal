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
// Forward declare the enum with its underlying type to avoid including the full protobuf header
enum Architecture : int;
class MeshGraphDescriptor;
class MeshDescriptor;
class GraphDescriptor;
class NodeRef;
class MeshRef;
class GraphRef;
enum TorusTopology_Type : int;
}

class MeshGraphDescriptor {

struct NodeInstance {
    const uint32_t id;
    const std::string descriptor_name;
    std::variant<const proto::MeshDescriptor*, const proto::GraphDescriptor*> descriptor;
    std::variant<const proto::MeshRef*, const proto::GraphRef*> ref;

    const uint32_t global_id = generate_next_global_id();

private:
    inline static uint32_t generate_next_global_id() {
        static std::atomic_uint32_t next_global_id_ = 0;
        return next_global_id_++;
    }
};

struct MeshInstance : public NodeInstance {
    const proto::Architecture arch;
    const std::vector<uint32_t> device_ids;
};

struct GraphInstance : public NodeInstance {
    std::string type;
    std::vector<std::shared_ptr<NodeInstance>> sub_instances = {};
};



public:
    explicit MeshGraphDescriptor(const std::string& text_proto);
    explicit MeshGraphDescriptor(const std::filesystem::path& text_proto_file_path);
    ~MeshGraphDescriptor();

    // Print functions for debugging and inspection
    void print_node_instance(const std::shared_ptr<NodeInstance>& node_instance, int indent_level = 0);
    void print_all_nodes();
    
    // Accessor methods - all lookups go through the global ID map
    std::shared_ptr<NodeInstance> get_instance_by_global_id(uint32_t global_id) const;
    std::vector<std::shared_ptr<NodeInstance>> get_instances_by_type(const std::string& type) const;
    std::vector<std::shared_ptr<NodeInstance>> get_instances_by_name(const std::string& name) const;
    std::vector<std::shared_ptr<NodeInstance>> get_all_instances() const;

private:
    std::unique_ptr<const proto::MeshGraphDescriptor> proto_;


    std::unordered_map<std::string, const proto::MeshDescriptor*> mesh_descriptors_by_name_;
    std::unordered_map<std::string, const proto::GraphDescriptor*> graph_descriptors_by_name_;
    std::unordered_map<std::string, std::unordered_set<const proto::GraphDescriptor*>> graph_descriptors_by_type_;

    // Single source of truth - all instances by global ID
    std::unordered_map<uint32_t, std::shared_ptr<NodeInstance>> all_instances_;
    
    // Secondary indices - store only global IDs for memory efficiency
    std::shared_ptr<NodeInstance> top_level_instance_;
    std::unordered_map<std::string, std::vector<uint32_t>> instances_by_type_;
    std::unordered_map<std::string, std::vector<uint32_t>> instances_by_name_;
    


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

    std::shared_ptr<NodeInstance> populate_instances(const proto::NodeRef& node_ref);
    std::shared_ptr<NodeInstance> construct_node_instance(const proto::NodeRef& node_ref);
    
    // Helper function to get instance type
    std::string get_instance_type(const std::shared_ptr<NodeInstance>& node_instance);
    

};

}  // namespace tt::tt_fabric
