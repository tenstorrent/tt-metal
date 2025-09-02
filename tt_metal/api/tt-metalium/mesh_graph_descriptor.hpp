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

    enum class NodeKind : uint8_t { Mesh = 0, Graph = 1 };

    struct NodeIndex {
        NodeKind kind;
        NodeId idx; // index into meshes_ or graphs_
    };

    // TODO: Contenplate making graph instance data the same as mesh Instance data and merging into one struct
    struct MeshInstanceData {
        const NodeId local_id;             // instance id from proto
        const std::string_view name;         // points into proto_ storage
        const proto::Architecture arch;
        const proto::MeshDescriptor* desc; // non-owning pointer into proto_
    };

    struct GraphInstanceData {
        const NodeId local_id;             // instance id from proto
        const std::string_view name;         // points into proto_ storage
        const std::string_view type;         // points into proto_ storage
        const proto::GraphDescriptor* desc; // non-owning pointer into proto_
        const std::unordered_set<NodeId> sub_instances; // direct list of child NodeIds // GLOBAL IDS
    };

    explicit MeshGraphDescriptor(const std::string& text_proto);
    explicit MeshGraphDescriptor(const std::filesystem::path& text_proto_file_path);
    ~MeshGraphDescriptor();

    // Debugging/inspection
    void print_node(NodeId id, int indent_level = 0);
    void print_all_nodes();

    // Cast-free API
    NodeId top_level() const { return top_level_id_; }
    bool is_mesh(NodeId id) const { return nodes_[id].kind == NodeKind::Mesh; }
    bool is_graph(NodeId id) const { return nodes_[id].kind == NodeKind::Graph; }
    const MeshInstanceData& mesh(NodeId id) const { return meshes_[nodes_[id].idx]; }
    const GraphInstanceData& graph(NodeId id) const { return graphs_[nodes_[id].idx]; }

    // Typed enumeration
    const std::vector<NodeId>& all_meshes() const { return mesh_ids_; }
    const std::vector<NodeId>& all_graphs() const { return graph_ids_; }

    // Queries
    const std::vector<NodeId>& by_name(std::string_view name) const { return name_to_ids_.at(name); }
    const std::vector<NodeId>& by_type(std::string_view type) const { return type_to_ids_.at(type); } // includes "mesh"

private:
    // Descriptor fast lookup
    std::unique_ptr<const proto::MeshGraphDescriptor> proto_;
    std::unordered_map<std::string_view, const proto::MeshDescriptor*> mesh_desc_by_name_;
    std::unordered_map<std::string_view, const proto::GraphDescriptor*> graph_desc_by_name_;

    // Global node table and typed stores
    std::vector<NodeIndex> nodes_;
    std::vector<MeshInstanceData> meshes_;
    std::vector<GraphInstanceData> graphs_;

    // Indices
    std::unordered_map<std::string_view, std::vector<NodeId>> name_to_ids_;
    std::unordered_map<std::string_view, std::vector<NodeId>> type_to_ids_;
    std::vector<NodeId> mesh_ids_;
    std::vector<NodeId> graph_ids_;
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
    NodeId populate_instance(const proto::NodeRef& node_ref);
    NodeId populate_mesh_instance(const proto::MeshRef& mesh_ref);
    NodeId populate_graph_instance(const proto::GraphRef& graph_ref);
    

};

}  // namespace tt::tt_fabric
