// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <google/protobuf/text_format.h>
#include <tt_stl/caseless_comparison.hpp>
#include <tt_stl/reflection.hpp>
#include <tt_stl/span.hpp>
#include <tt_stl/strong_type.hpp>
#include <enchantum/enchantum.hpp>
#include <scaleout_tools/board/board.hpp>

// Add protobuf includes
#include "cluster_config.pb.h"
#include "pod_config.pb.h"
#include "deployment.pb.h"

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
inline std::ostream& operator<<(std::ostream& os, const PhysicalChannelConnection& conn) {
    os << "PhysicalChannelConnection{hostname='" << conn.hostname << "', tray_id=" << *conn.tray_id
       << ", asic_location=" << conn.asic_location << ", channel_id=" << *conn.channel_id << "}";
    return os;
}
inline std::ostream& operator<<(std::ostream& os, const PhysicalPortConnection& conn) {
    os << "PhysicalPortConnection{hostname='" << conn.hostname << "', aisle='" << conn.aisle << "', rack=" << conn.rack
       << ", shelf_u=" << conn.shelf_u << ", port_type=" << enchantum::to_string(conn.port_type)
       << ", port_id=" << *conn.port_id << "}";
    return os;
}

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
    // Helper to load protobuf descriptors
    template <typename Descriptor>
    static Descriptor load_descriptor_from_textproto(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_path);
        }

        std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        Descriptor descriptor;
        if (!google::protobuf::TextFormat::ParseFromString(file_content, &descriptor)) {
            throw std::runtime_error("Failed to parse textproto file: " + file_path);
        }
        return descriptor;
    }

    // Constructor
    CablingGenerator(const std::string& cluster_descriptor_path, const std::string& deployment_descriptor_path);

    // Getters for all data
    const std::vector<Host>& get_deployment_hosts() const;
    const std::unordered_map<std::pair<HostId, TrayId>, const Board*, HostTrayHasher>& get_boards_by_host_tray() const;
    const std::vector<LogicalChipConnectionPair>& get_chip_connections() const;

    // Method to emit factory system descriptor
    void emit_factory_system_descriptor(const std::string& output_path) const;

private:
    // Find pod descriptor by name - search inline first, then fallback to file
    tt::scaleout_tools::cabling_generator::proto::PodDescriptor find_pod_descriptor(
        const std::string& pod_descriptor_name,
        const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor);

    // Build pod from descriptor with port connections and validation
    Pod build_pod(
        const std::string& pod_descriptor_name,
        HostId host_id,
        const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor);

    // Build resolved graph instance from template and concrete host mappings
    std::shared_ptr<ResolvedGraphInstance> build_graph_instance(
        const tt::scaleout_tools::cabling_generator::proto::GraphInstance& graph_instance,
        const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
        const tt::scaleout_tools::deployment::proto::DeploymentDescriptor& deployment_descriptor,
        const std::string& instance_name = "");

    // Build cluster from descriptor with port connections and validation
    void build_cluster_from_descriptor(
        const tt::scaleout_tools::cabling_generator::proto::ClusterDescriptor& cluster_descriptor,
        const tt::scaleout_tools::deployment::proto::DeploymentDescriptor& deployment_descriptor);

    // Validate that each host_id is assigned to exactly one pod
    void validate_host_id_uniqueness();

    // Recursively collect all host_id assignments with their pod paths
    void collect_host_assignments(
        std::shared_ptr<ResolvedGraphInstance> graph,
        const std::string& path_prefix,
        std::unordered_map<HostId, std::string>& host_to_pod_path);

    // Simple path resolution for connection processing
    std::pair<Pod&, HostId> resolve_pod_from_path(
        ttsl::Span<const std::string> path, std::shared_ptr<ResolvedGraphInstance> graph = nullptr);

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
struct hash<tt::scaleout_tools::LogicalChipConnection> {
    std::size_t operator()(const tt::scaleout_tools::LogicalChipConnection& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(
            *conn.host_id, conn.tray_id, conn.asic_channel.asic_location, conn.asic_channel.channel_id);
    }
};

template <>
struct hash<tt::scaleout_tools::PhysicalChannelConnection> {
    std::size_t operator()(const tt::scaleout_tools::PhysicalChannelConnection& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(
            conn.hostname, *conn.tray_id, conn.asic_location, conn.channel_id);
    }
};

template <>
struct hash<tt::scaleout_tools::PhysicalPortConnection> {
    std::size_t operator()(const tt::scaleout_tools::PhysicalPortConnection& conn) const {
        return tt::stl::hash::hash_objects_with_default_seed(
            conn.hostname, conn.aisle, conn.rack, conn.shelf_u, conn.port_type, *conn.port_id);
    }
};

}  // namespace std
