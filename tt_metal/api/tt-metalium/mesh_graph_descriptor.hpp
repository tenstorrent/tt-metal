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

#include "assert.hpp"

// Forward declaration
namespace tt::tt_fabric {

namespace proto {
// Forward declare the enum with its underlying type to avoid including the full protobuf header
enum Architecture : int;
class MeshGraphDescriptor;
class MeshDescriptor;
class GraphDescriptor;
class Channels;
class NodeRef;
class MeshRef;
class GraphRef;
enum Policy : int;
}

// TODO: Try make efficient by storing stringviews?
class MeshGraphDescriptor {
public:
    using LocalNodeId = uint32_t;   // Scoped to parent (mesh_id, graph_id, device index)
    using GlobalNodeId = uint32_t;  // Unique across the instantiated MGD
    using ConnectionId = uint32_t;

    enum class NodeKind : uint8_t { Mesh = 0, Graph = 1, Device = 2 };

    // NOTE: Instance Data and ConnectionData are subject to change as Physical discovery is implemented
    struct InstanceData {
        const LocalNodeId local_id;        // instance id from proto or computed device index
        const std::string name;
        const std::string type;         // points into proto_ storage
        const NodeKind kind;
        std::variant<const proto::MeshDescriptor*, const proto::GraphDescriptor*> desc;
        std::unordered_set<GlobalNodeId> sub_instances; // direct list of child GlobalNodeIds
        std::unordered_map<LocalNodeId, GlobalNodeId> sub_instances_local_id_to_global_id; // child LocalId -> GlobalId
        std::vector<GlobalNodeId> hierarchy; // path from root using GlobalNodeIds

        const GlobalNodeId global_id = generate_next_global_id();

    private:
        inline static GlobalNodeId generate_next_global_id() {
            static std::atomic_uint32_t next_global_id_ = 0;
            return next_global_id_++;
        }
    };

    struct ConnectionData {
        std::vector<GlobalNodeId> nodes; // [src_global_device_id, dst_global_device_id]
        const std::uint32_t count;             // ethernet lanes per connection
        const proto::Policy policy;
        const bool directional;
        const GlobalNodeId parent_instance_id;

        const ConnectionId connection_id = generate_next_global_id();

        private:
            inline static ConnectionId generate_next_global_id() {
                static std::atomic_uint32_t next_global_id_ = 0;
                return next_global_id_++;
            }
    };

    explicit MeshGraphDescriptor(const std::string& text_proto,  const bool backwards_compatible = false);
    explicit MeshGraphDescriptor(const std::filesystem::path& text_proto_file_path,  const bool backwards_compatible = false);
    ~MeshGraphDescriptor();

    // Debugging/inspection
    void print_node(GlobalNodeId id, int indent_level = 0);
    void print_all_nodes();

    const InstanceData & top_level() const {
        auto it = instances_.find(top_level_id_);
        TT_FATAL(it != instances_.end(), "Top-level instance id {} not found", top_level_id_);
        return it->second;
    }
    bool is_graph(const InstanceData& instance) const { return instance.kind == NodeKind::Graph; }
    bool is_mesh(const InstanceData& instance) const { return instance.kind == NodeKind::Mesh; }
    const InstanceData& get_instance(GlobalNodeId id) const {
        auto it = instances_.find(id);
        TT_FATAL(it != instances_.end(), "Instance id {} not found", id);
        return it->second;
    }
    const ConnectionData& get_connection(ConnectionId connection_id) const {
        auto it = connections_.find(connection_id);
        TT_FATAL(it != connections_.end(), "Connection id {} not found", connection_id);
        return it->second;
    }


    // Typed enumeration
    const std::vector<GlobalNodeId>& all_meshes() const { return mesh_instances_; }
    const std::vector<GlobalNodeId>& all_graphs() const { return graph_instances_; }

    // Queries
    const std::vector<GlobalNodeId>& instances_by_name(std::string name) const {
        auto it = instances_by_name_.find(name);
        TT_FATAL(it != instances_by_name_.end(), "No instances found with name {}", name);
        return it->second;
    }
    const std::vector<GlobalNodeId>& instances_by_type(std::string type) const { // includes "MESH"
        auto it = instances_by_type_.find(type);
        TT_FATAL(it != instances_by_type_.end(), "No instances found with type {}", type);
        return it->second;
    }
    const std::vector<ConnectionId>& connections_by_instance_id(GlobalNodeId instance_id) const {
        auto it = connections_by_instance_id_.find(instance_id);
        TT_FATAL(it != connections_by_instance_id_.end(), "No connections indexed for instance id {}", instance_id);
        return it->second;
    }
    const std::vector<ConnectionId>& connections_by_type(std::string type) const {
        auto it = connections_by_type_.find(type);
        TT_FATAL(it != connections_by_type_.end(), "No connections found for type {}", type);
        return it->second;
    }
    const std::vector<ConnectionId>& connections_by_source_device_id(GlobalNodeId source_device_id) const {
        auto it = connections_by_source_device_id_.find(source_device_id);
        TT_FATAL(it != connections_by_source_device_id_.end(), "No connections found for source device id {}", source_device_id);
        return it->second;
    }


private:
    const bool backwards_compatible_;

    // Descriptor fast lookup
    std::unique_ptr<const proto::MeshGraphDescriptor> proto_;
    std::unordered_map<std::string_view, const proto::MeshDescriptor*> mesh_desc_by_name_;
    std::unordered_map<std::string_view, const proto::GraphDescriptor*> graph_desc_by_name_;

    // Global node table and typed stores
    std::unordered_map<GlobalNodeId, InstanceData> instances_;
    std::unordered_map<ConnectionId, ConnectionData> connections_;

    // Indices
    std::unordered_map<std::string, std::vector<GlobalNodeId>> instances_by_name_;
    std::unordered_map<std::string_view, std::vector<GlobalNodeId>> instances_by_type_;
    std::vector<GlobalNodeId> device_instances_;
    std::vector<GlobalNodeId> mesh_instances_;
    std::vector<GlobalNodeId> graph_instances_;
    GlobalNodeId top_level_id_;

    // Connections
    std::unordered_map<GlobalNodeId, std::vector<ConnectionId>> connections_by_instance_id_;
    std::unordered_map<std::string, std::vector<ConnectionId>> connections_by_type_;
    std::unordered_map<GlobalNodeId, std::vector<ConnectionId>> connections_by_source_device_id_;


    static void set_defaults(proto::MeshGraphDescriptor& proto);
    static std::vector<std::string> static_validate(const proto::MeshGraphDescriptor& proto, const bool backwards_compatible = false);

    // Helper methods for validation that return their own error lists
    static void validate_basic_structure(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& errors);
    static void validate_names(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& errors);
    static void validate_mesh_topology(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& errors);
    static void validate_architecture_consistency(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& errors);
    static void validate_channels(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& errors);
    static void validate_express_connections(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& errors);
    static void validate_graph_descriptors(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& errors);
    static void validate_graph_topology_and_connections(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& errors);

    static void validate_legacy_requirements(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& errors);

    void populate();

    // Populate Descriptors
    void populate_descriptors();

    // Populate Instances
    GlobalNodeId populate_instance(const proto::NodeRef& node_ref, std::vector<GlobalNodeId>& hierarchy);
    GlobalNodeId populate_mesh_instance(const proto::MeshRef& mesh_ref, std::vector<GlobalNodeId>& hierarchy);
    GlobalNodeId populate_graph_instance(const proto::GraphRef& graph_ref, std::vector<GlobalNodeId>& hierarchy);
    GlobalNodeId populate_device_instance(LocalNodeId local_id, std::vector<GlobalNodeId>& hierarchy);

    // Populate Connections
    void populate_connections();
    void add_to_fast_lookups(const InstanceData& instance);
    void add_connection_to_fast_lookups(const ConnectionData& connection, const std::string& type);

    void populate_intra_mesh_connections(GlobalNodeId mesh_id);
    void populate_intra_mesh_express_connections(GlobalNodeId mesh_id);
    void populate_inter_mesh_connections(GlobalNodeId graph_id);
    void populate_inter_mesh_manual_connections(GlobalNodeId graph_id);
    void populate_inter_mesh_topology_connections(GlobalNodeId graph_id); // TODO: To be implemented in seperate PR

    GlobalNodeId find_instance_by_ref(GlobalNodeId parent_instance_id, const proto::NodeRef& node_ref);

};

}  // namespace tt::tt_fabric
