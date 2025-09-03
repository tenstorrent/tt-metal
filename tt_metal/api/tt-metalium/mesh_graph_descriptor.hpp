// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <string_view>
#include <filesystem>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <variant>
#include <atomic>

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
public:
    using NodeId = uint32_t;

    enum class NodeKind : uint8_t { Mesh = 0, Graph = 1, Device = 2 };

    struct InstanceData {
        const NodeId local_id;             // instance id from proto
        const std::string name;
        const std::string_view type;         // points into proto_ storage
        const NodeKind kind;
        std::variant<const proto::MeshDescriptor*, const proto::GraphDescriptor*> desc;
        std::unordered_set<NodeId> sub_instances; // direct list of child NodeIds // GLOBAL IDS
        std::unordered_map<NodeId, NodeId> sub_instances_local_id_to_global_id; // map of child NodeIds to child NodeIds // GLOBAL IDS
        std::vector<NodeId> hierarchy;

        const NodeId global_id = generate_next_global_id();

    private:
        inline static uint32_t generate_next_global_id() {
            static std::atomic_uint32_t next_global_id_ = 0;
            return next_global_id_++;
        }
    };

    using ConnectionNodeRef = std::vector<NodeId>;

    struct ConnnectionData {};
    explicit MeshGraphDescriptor(const std::string& text_proto);
    explicit MeshGraphDescriptor(const std::filesystem::path& text_proto_file_path);
    ~MeshGraphDescriptor();

    // Debugging/inspection
    void print_node(NodeId id, int indent_level = 0);
    void print_all_nodes();

    const InstanceData & top_level() const { return instances_.at(top_level_id_); }
    bool is_graph(const InstanceData& instance) const { return instance.kind == NodeKind::Graph; }
    bool is_mesh(const InstanceData& instance) const { return instance.kind == NodeKind::Mesh; }
    const InstanceData& at(NodeId id) const { return instances_.at(id); }

    // Typed enumeration
    const std::vector<NodeId>& all_meshes() const { return mesh_instances_; }
    const std::vector<NodeId>& all_graphs() const { return graph_instances_; }

    // Queries
    const std::vector<NodeId>& by_name(std::string_view name) const { return instances_by_name_.at(name); }
    const std::vector<NodeId>& by_type(std::string_view type) const { return instances_by_type_.at(type); } // includes "MESH"

private:
    // Descriptor fast lookup
    std::unique_ptr<const proto::MeshGraphDescriptor> proto_;
    std::unordered_map<std::string_view, const proto::MeshDescriptor*> mesh_desc_by_name_;
    std::unordered_map<std::string_view, const proto::GraphDescriptor*> graph_desc_by_name_;

    // Global node table and typed stores
    std::unordered_map<NodeId, InstanceData> instances_;

    // Indices
    std::unordered_map<std::string_view, std::vector<NodeId>> instances_by_name_;
    std::unordered_map<std::string_view, std::vector<NodeId>> instances_by_type_;
    std::vector<NodeId> mesh_instances_;
    std::vector<NodeId> graph_instances_;
    NodeId top_level_id_;
    
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
    void populate_descriptors();
    NodeId populate_instance(const proto::NodeRef& node_ref, std::vector<NodeId>& hierarchy);
    NodeId populate_mesh_instance(const proto::MeshRef& mesh_ref, std::vector<NodeId>& hierarchy);
    NodeId populate_graph_instance(const proto::GraphRef& graph_ref, std::vector<NodeId>& hierarchy);
    

};

}  // namespace tt::tt_fabric
