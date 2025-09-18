// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include <tt_stl/strong_type.hpp>
#include <board/board.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::scaleout_tools {

// Strong types to prevent mixing with other uint32_t values
using HostId = ttsl::StrongType<uint32_t, struct HostIdTag>;
using TrayId = ttsl::StrongType<uint32_t, struct TrayIdTag>;

struct Host {
    std::string hostname;
    std::string hall;
    std::string aisle;
    uint32_t rack = 0;
    uint32_t shelf_u = 0;
    std::string motherboard;
};

struct LogicalChannelEndpoint {
    HostId host_id{0};
    TrayId tray_id{0};
    AsicChannel asic_channel;

    auto operator<=>(const LogicalChannelEndpoint& other) const = default;
};

struct PhysicalChannelEndpoint {
    std::string hostname;
    TrayId tray_id{0};
    AsicChannel asic_channel;

    auto operator<=>(const PhysicalChannelEndpoint& other) const = default;
};

struct PhysicalPortEndpoint {
    std::string hostname;
    std::string aisle;
    uint32_t rack = 0;
    uint32_t shelf_u = 0;
    TrayId tray_id{0};
    PortType port_type = PortType::TRACE;
    PortId port_id{0};

    auto operator<=>(const PhysicalPortEndpoint& other) const = default;
};

// Overload operator<< for readable test output
std::ostream& operator<<(std::ostream& os, const PhysicalChannelEndpoint& conn);
std::ostream& operator<<(std::ostream& os, const PhysicalPortEndpoint& conn);

using LogicalChannelConnection = std::pair<LogicalChannelEndpoint, LogicalChannelEndpoint>;
using PhysicalChannelConnection = std::pair<PhysicalChannelEndpoint, PhysicalChannelEndpoint>;

struct Node {
    std::string motherboard;
    std::map<TrayId, Board> boards;
    HostId host_id{0};
    // Board-to-board connections within this node: PortType -> [(tray_id, port_id) <-> (tray_id, port_id)]
    using PortEndpoint = std::pair<TrayId, PortId>;
    using PortConnection = std::pair<PortEndpoint, PortEndpoint>;
    std::unordered_map<PortType, std::vector<PortConnection>> inter_board_connections;
};

// Resolved graph instance with concrete nodes
struct ResolvedGraphInstance {
    std::string template_name;
    std::string instance_name;
    std::map<std::string, Node> nodes;                                        // Direct node children (by name)
    std::map<std::string, std::unique_ptr<ResolvedGraphInstance>> subgraphs;  // Nested graph children

    // All connections within this graph instance
    using PortEndpoint = std::tuple<HostId, TrayId, PortId>;  // host_id, tray_id, port_id
    using PortConnection = std::pair<PortEndpoint, PortEndpoint>;
    std::unordered_map<PortType, std::vector<PortConnection>> internal_connections;
};

enum class CableLength { CABLE_0P5, CABLE_1, CABLE_2P5, CABLE_3, CABLE_5, UNKNOWN };

CableLength calc_cable_length(const Host& host1, const Host& host2);

class CablingGenerator {
public:
    // Constructor
    CablingGenerator(const std::string& cluster_descriptor_path, const std::string& deployment_descriptor_path);

    // Getters for all data
    const std::vector<Host>& get_deployment_hosts() const;
    const std::vector<LogicalChannelConnection>& get_chip_connections() const;

    // Method to emit factory system descriptor
    void emit_factory_system_descriptor(const std::string& output_path) const;

    // Method to emit cabling guide CSV
    void emit_cabling_guide_csv(const std::string& output_path, bool loc_info = true) const;

private:
    // Validate that each host_id is assigned to exactly one node
    void validate_host_id_uniqueness();

    // Recursively collect all host_id assignments with their node paths
    void collect_host_assignments(
        const std::unique_ptr<ResolvedGraphInstance>& graph,
        const std::string& path_prefix,
        std::unordered_map<HostId, std::string>& host_to_node_path);

    // Utility function to generate logical chip connections from cluster hierarchy
    void generate_logical_chip_connections();

    void generate_connections_from_resolved_graph(const std::unique_ptr<ResolvedGraphInstance>& graph);

    void populate_host_id_to_node();

    void populate_host_id_from_resolved_graph(const std::unique_ptr<ResolvedGraphInstance>& graph);

    void get_all_connections_of_type(
        const std::unique_ptr<ResolvedGraphInstance>& instance,
        const std::vector<PortType>& port_types,
        std::vector<std::pair<std::tuple<HostId, TrayId, PortId>, std::tuple<HostId, TrayId, PortId>>>& conn_list)
        const;

    // Caches for optimization
    std::unordered_map<tt::umd::BoardType, Board> board_templates_;
    std::unordered_map<std::string, Node> node_templates_;  // Templates with host_id=0

    std::unique_ptr<ResolvedGraphInstance> root_instance_;
    std::map<HostId, Node*> host_id_to_node_;  // Global lookup map for HostId -> Node reference
    // Guaranteed to be sorted
    std::vector<LogicalChannelConnection> chip_connections_;
    std::vector<Host> deployment_hosts_;
};

}  // namespace tt::scaleout_tools

// Hash specializations
namespace std {
template <>
struct hash<tt::scaleout_tools::LogicalChannelEndpoint>;

template <>
struct hash<tt::scaleout_tools::PhysicalChannelEndpoint>;

template <>
struct hash<tt::scaleout_tools::PhysicalPortEndpoint>;

}  // namespace std
