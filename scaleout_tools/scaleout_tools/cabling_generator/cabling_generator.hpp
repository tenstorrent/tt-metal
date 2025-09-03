// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <utility>

#include <tt_stl/strong_type.hpp>
#include <scaleout_tools/board/board.hpp>

namespace tt::scaleout_tools {

// Strong types to prevent mixing with other uint32_t values
using HostId = ttsl::StrongType<uint32_t, struct HostIdTag>;
using TrayId = ttsl::StrongType<uint32_t, struct TrayIdTag>;

// Custom hasher for (HostId, TrayId) pairs
struct HostTrayHasher {
    std::size_t operator()(const std::pair<HostId, TrayId>& p) const {
        return std::hash<uint64_t>{}((static_cast<uint64_t>(*p.first) << 32) | *p.second);
    }
};

struct Host {
    std::string hostname;
    std::string hall;
    std::string aisle;
    uint32_t rack;
    uint32_t shelf_u;
};

struct LogicalChipConnection {
    HostId host_id;
    TrayId tray_id;
    AsicChannel asic_channel;

    auto operator<=>(const LogicalChipConnection& other) const = default;
};

struct PhysicalChannelConnection {
    std::string hostname;
    TrayId tray_id;
    uint32_t asic_location;
    tt::scaleout_tools::ChanId channel_id;

    auto operator<=>(const PhysicalChannelConnection& other) const = default;
};

struct PhysicalPortConnection {
    std::string hostname;
    std::string aisle;
    uint32_t rack;
    uint32_t shelf_u;
    PortType port_type;
    PortId port_id;

    auto operator<=>(const PhysicalPortConnection& other) const = default;
};

// Overload operator<< for readable test output
std::ostream& operator<<(std::ostream& os, const PhysicalChannelConnection& conn);
std::ostream& operator<<(std::ostream& os, const PhysicalPortConnection& conn);

using LogicalChipConnectionPair = std::pair<LogicalChipConnection, LogicalChipConnection>;

struct Pod {
    std::unordered_map<TrayId, Board> boards;
    HostId host_id = HostId(0);
    // Board-to-board connections within this pod: PortType -> [(tray_id, port_id) <-> (tray_id, port_id)]
    std::unordered_map<PortType, std::vector<std::pair<std::pair<TrayId, PortId>, std::pair<TrayId, PortId>>>>
        inter_board_connections;
};

// Resolved graph instance with concrete pods
struct ResolvedGraphInstance {
    std::string template_name;
    std::string instance_name;
    std::unordered_map<std::string, Pod> pods;                                          // Direct pod children (by name)
    std::unordered_map<std::string, std::shared_ptr<ResolvedGraphInstance>> subgraphs;  // Nested graph children

    // All connections within this graph instance
    std::unordered_map<
        PortType,
        std::vector<std::pair<
            std::tuple<std::vector<std::string>, TrayId, PortId>,  // Path, tray_id, port_id
            std::tuple<std::vector<std::string>, TrayId, PortId>>>>
        internal_connections;
};

class CablingGenerator {
public:
    // Constructor
    CablingGenerator(const std::string& cluster_descriptor_path, const std::string& deployment_descriptor_path);

    // Getters for all data
    const std::vector<Host>& get_deployment_hosts() const;
    const std::unordered_map<std::pair<HostId, TrayId>, const Board*, HostTrayHasher>& get_boards_by_host_tray() const;
    const std::vector<LogicalChipConnectionPair>& get_chip_connections() const;

    // Method to emit factory system descriptor
    void emit_factory_system_descriptor(const std::string& output_path) const;

private:
    // Validate that each host_id is assigned to exactly one pod
    void validate_host_id_uniqueness();

    // Recursively collect all host_id assignments with their pod paths
    void collect_host_assignments(
        std::shared_ptr<ResolvedGraphInstance> graph,
        const std::string& path_prefix,
        std::unordered_map<HostId, std::string>& host_to_pod_path);

    // Utility function to generate logical chip connections from cluster hierarchy
    void generate_logical_chip_connections();

    void generate_connections_from_resolved_graph(std::shared_ptr<ResolvedGraphInstance> graph);

    void populate_boards_by_host_tray();

    void populate_boards_from_resolved_graph(std::shared_ptr<ResolvedGraphInstance> graph);

    // Caches for optimization
    std::unordered_map<std::string, Board> board_templates_;
    std::unordered_map<std::string, Pod> pod_templates_;  // Templates with host_id=0

    std::shared_ptr<ResolvedGraphInstance> root_instance_;
    std::unordered_map<std::pair<HostId, TrayId>, const Board*, HostTrayHasher> boards_by_host_tray_;
    std::vector<LogicalChipConnectionPair> chip_connections_;
    std::vector<Host> deployment_hosts_;
};

}  // namespace tt::scaleout_tools

// Hash specializations
namespace std {
template <>
struct hash<tt::scaleout_tools::LogicalChipConnection>;

template <>
struct hash<tt::scaleout_tools::PhysicalChannelConnection>;

template <>
struct hash<tt::scaleout_tools::PhysicalPortConnection>;

}  // namespace std
