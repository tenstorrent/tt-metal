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

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

// Forward declaration
namespace tt::tt_fabric {

namespace proto {
// Forward declare the enum with its underlying type to avoid including the full protobuf header
enum Architecture : int;
class MeshGraphDescriptor;
class MeshDescriptor;
class GraphDescriptor;
class SwitchDescriptor;
class Channels;
class NodeRef;
class MeshRef;
class GraphRef;
class SwitchRef;
enum Policy : int;
enum RoutingDirection : int;
class LogicalFabricNodeId;
class PhysicalAsicPosition;
class AsicPinning;
}  // namespace proto

inline namespace v1_1 {
using LocalNodeId = uint32_t;   // Scoped to parent (mesh_id, graph_id, device index)
using GlobalNodeId = uint32_t;  // Unique across the instantiated MGD
using ConnectionId = uint32_t;

enum class NodeKind : uint8_t { Mesh = 0, Graph = 1, Device = 2, Switch = 3 };
// NOTE: Instance Data and ConnectionData are subject to change as Physical discovery is implemented
// These will be moved to Mesh Graph object once MGD 1.0 is deprecated
struct InstanceData {
    LocalNodeId local_id;  // instance id from proto or computed device index
    std::string name;
    std::string type;
    NodeKind kind;  // Type of instance (mesh, graph, device, switch)
    std::variant<const proto::MeshDescriptor*, const proto::GraphDescriptor*, const proto::SwitchDescriptor*>
        desc;                                        // Pointer to the descriptor that this instance is based on
    std::unordered_set<GlobalNodeId> sub_instances;  // direct list of child GlobalNodeIds
    std::unordered_map<LocalNodeId, GlobalNodeId> sub_instances_local_id_to_global_id;  // child LocalId -> GlobalId
    std::vector<GlobalNodeId> hierarchy;  // path from root using GlobalNodeIds

    GlobalNodeId global_id = generate_next_global_id();

private:
    static GlobalNodeId generate_next_global_id() {
        static std::atomic_uint32_t next_global_id_ = 0;
        return next_global_id_++;
    }
};

struct ConnectionData {
    std::vector<GlobalNodeId> nodes;  // [src_global_device_id, dst_global_device_id]
    std::uint32_t count;              // ethernet lanes per connection
    proto::Policy policy;
    GlobalNodeId parent_instance_id;

    ConnectionId connection_id = generate_next_global_id();

    // TODO: Remove after MGD 1.0 is deprecated
    proto::RoutingDirection routing_direction;

    // Flag to assign Z direction to intermesh connections (dev/testing feature)
    bool assign_z_direction = false;

private:
    static ConnectionId generate_next_global_id() {
        static std::atomic_uint32_t next_global_id_ = 0;
        return next_global_id_++;
    }
};
}  // namespace v1_1

// FabricNodeId is now defined in fabric_types.hpp (already included above)

// Use ASICPosition type alias for consistency with TopologyMapper
using AsicPosition = tt::tt_metal::ASICPosition;

// TODO: Try make efficient by storing stringviews?
class MeshGraphDescriptor {
public:
    // backwards_compatible will enable all checks related to MGD 1.0. This will limit the functionality of MGD 2.0
    explicit MeshGraphDescriptor(const std::string& text_proto, bool backwards_compatible = false);
    explicit MeshGraphDescriptor(const std::filesystem::path& text_proto_file_path, bool backwards_compatible = false);

    ~MeshGraphDescriptor();

    // Debugging/inspection
    void print_node(GlobalNodeId id, int indent_level = 0);
    void print_all_nodes();

    // Instance access API
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
    const InstanceData& top_level() const {
        auto it = instances_.find(top_level_id_);
        TT_FATAL(it != instances_.end(), "Top-level instance id {} not found", top_level_id_);
        return it->second;
    }

    // Instance type checks
    bool is_graph(const InstanceData& instance) const { return instance.kind == NodeKind::Graph; }
    bool is_mesh(const InstanceData& instance) const { return instance.kind == NodeKind::Mesh; }
    bool is_switch(const InstanceData& instance) const { return instance.kind == NodeKind::Switch; }

    // Typed enumeration
    const std::vector<GlobalNodeId>& all_meshes() const { return mesh_instances_; }
    const std::vector<GlobalNodeId>& all_graphs() const { return graph_instances_; }
    const std::vector<GlobalNodeId>& all_switches() const { return switch_instances_; }

    // Queries
    const std::vector<GlobalNodeId>& instances_by_name(const std::string& name) const {
        auto it = instances_by_name_.find(name);
        TT_FATAL(it != instances_by_name_.end(), "No instances found with name: {}", name);
        return it->second;
    }
    const std::vector<GlobalNodeId>& instances_by_type(const std::string& type) const {  // includes "MESH"
        auto it = instances_by_type_.find(type);
        TT_FATAL(it != instances_by_type_.end(), "No instances found with type: {}", type);
        return it->second;
    }
    const std::vector<ConnectionId>& connections_by_instance_id(const GlobalNodeId instance_id) const {
        auto it = connections_by_instance_id_.find(instance_id);
        TT_FATAL(it != connections_by_instance_id_.end(), "No connections found for instance id: {}", instance_id);
        return it->second;
    }
    const std::vector<ConnectionId>& connections_by_type(const std::string& type) const {
        auto it = connections_by_type_.find(type);
        TT_FATAL(it != connections_by_type_.end(), "No connections found with type: {}", type);
        return it->second;
    }
    bool has_connections_of_type(const std::string& type) const { return connections_by_type_.contains(type); }
    const std::vector<ConnectionId>& connections_by_source_device_id(const GlobalNodeId source_device_id) const {
        auto it = connections_by_source_device_id_.find(source_device_id);
        TT_FATAL(
            it != connections_by_source_device_id_.end(),
            "No connections found for source device id: {}",
            source_device_id);
        return it->second;
    }

    // TODO: This will disappear after we move to Physical discovery
    proto::Architecture get_arch() const;
    uint32_t get_num_eth_ports_per_direction() const;

    // Helper to infer FabricType from MGD dim_types
    static FabricType infer_fabric_type_from_dim_types(const proto::MeshDescriptor* mesh_desc);

    const std::vector<std::pair<AsicPosition, FabricNodeId>>& get_pinnings() const { return pinnings_; }

private:
    // Descriptor fast lookup
    std::shared_ptr<const proto::MeshGraphDescriptor> proto_;
    std::unordered_map<std::string, const proto::MeshDescriptor*> mesh_desc_by_name_;
    std::unordered_map<std::string, const proto::GraphDescriptor*> graph_desc_by_name_;
    std::unordered_map<std::string, const proto::SwitchDescriptor*> switch_desc_by_name_;

    // Global node table and typed stores
    std::unordered_map<GlobalNodeId, InstanceData> instances_;
    std::unordered_map<ConnectionId, ConnectionData> connections_;

    // Indices
    std::unordered_map<std::string, std::vector<GlobalNodeId>> instances_by_name_;
    std::unordered_map<std::string, std::vector<GlobalNodeId>> instances_by_type_;
    std::vector<GlobalNodeId> device_instances_;
    std::vector<GlobalNodeId> mesh_instances_;
    std::vector<GlobalNodeId> graph_instances_;
    std::vector<GlobalNodeId> switch_instances_;
    GlobalNodeId top_level_id_;

    // Connections
    std::unordered_map<GlobalNodeId, std::vector<ConnectionId>> connections_by_instance_id_;
    std::unordered_map<std::string_view, std::vector<ConnectionId>> connections_by_type_;
    std::unordered_map<GlobalNodeId, std::vector<ConnectionId>> connections_by_source_device_id_;

    std::vector<std::pair<AsicPosition, FabricNodeId>> pinnings_;

    static void set_defaults(proto::MeshGraphDescriptor& proto);
    static std::vector<std::string> static_validate(
        const proto::MeshGraphDescriptor& proto, bool backwards_compatible = false);

    // Helper methods for validation that return their own error lists
    static void validate_basic_structure(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& errors);
    static void validate_names(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages);
    static void validate_mesh_topology(
        const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages);
    static void validate_architecture_consistency(
        const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages);
    static void validate_channels(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages);
    static void validate_express_connections(
        const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages);
    static void validate_switch_descriptors(
        const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages);
    static void validate_graph_descriptors(
        const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages);
    static void validate_graph_topology_and_connections(
        const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages);
    static void validate_pinnings(const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages);

    static void validate_legacy_requirements(
        const proto::MeshGraphDescriptor& proto, std::vector<std::string>& error_messages);

    // Populates the MGD Graph from the proto file
    void populate();

    // Populate Descriptors
    void populate_descriptors();

    // Populate Pinnings
    void populate_pinnings();

    // Populate Instances
    void populate_top_level_instance();
    GlobalNodeId populate_instance(const proto::NodeRef& node_ref, std::vector<GlobalNodeId>& hierarchy);
    GlobalNodeId populate_mesh_instance(const proto::MeshRef& mesh_ref, std::vector<GlobalNodeId>& hierarchy);
    GlobalNodeId populate_graph_instance(const proto::GraphRef& graph_ref, std::vector<GlobalNodeId>& hierarchy);
    GlobalNodeId populate_switch_instance(const proto::SwitchRef& switch_ref, std::vector<GlobalNodeId>& hierarchy);
    GlobalNodeId populate_device_instance(LocalNodeId local_id, std::vector<GlobalNodeId>& hierarchy);

    // Populate Connections
    void populate_connections();

    void pre_populate_connections_lookups();

    void populate_intra_mesh_connections(GlobalNodeId mesh_id);
    void populate_intra_mesh_express_connections(GlobalNodeId mesh_id);
    void populate_inter_mesh_connections(GlobalNodeId graph_id);
    void populate_inter_mesh_manual_connections(GlobalNodeId graph_id);
    void populate_inter_mesh_topology_connections(GlobalNodeId graph_id);  // TODO: To be implemented in seperate PR
    void populate_inter_mesh_topology_connections_all_to_all(GlobalNodeId graph_id);
    void populate_inter_mesh_topology_connections_ring(GlobalNodeId graph_id);

    GlobalNodeId find_instance_by_ref(GlobalNodeId parent_instance_id, const proto::NodeRef& node_ref);

    void add_to_fast_lookups(const InstanceData& instance);
    void add_connection_to_fast_lookups(const ConnectionData& connection, const std::string& type);
};

}  // namespace tt::tt_fabric
