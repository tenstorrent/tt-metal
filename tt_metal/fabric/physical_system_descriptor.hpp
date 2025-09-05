// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <umd/device/types/cluster_descriptor_types.h>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal {

using AsicID = tt::stl::StrongType<uint64_t, struct AsicIDTag>;
using TrayID = tt::stl::StrongType<uint32_t, struct TrayIDTag>;
using ASICLocation = tt::stl::StrongType<uint32_t, struct ASICLocationTag>;
using RackID = tt::stl::StrongType<uint32_t, struct RackIDTag>;
using UID = tt::stl::StrongType<uint32_t, struct UIDTag>;
using HallID = tt::stl::StrongType<uint32_t, struct HallIDTag>;
using AisleID = tt::stl::StrongType<uint32_t, struct AisleIDTag>;

// Specify Physical ASIC Attributes
struct ASICDescriptor {
    TrayID tray_id;
    ASICLocation asic_location;
    BoardType board_type = BoardType::UNKNOWN;
    AsicID unique_id;
    std::string host_name;
};

// Specify an ethernet connection between two ASICs
struct EthConnection {
    uint8_t src_chan = 0;
    uint8_t dst_chan = 0;
    bool is_local = false;

    bool operator==(const EthConnection& other) const {
        return src_chan == other.src_chan && dst_chan == other.dst_chan && other.is_local == is_local;
    }
    bool operator<(const EthConnection& other) const {
        if (src_chan != other.src_chan) {
            return src_chan < other.src_chan;
        }
        return dst_chan < other.dst_chan;
    }
};

// Specify an ethernet connection between two hosts
struct ExitNodeConnection {
    AsicID src_exit_node;
    AsicID dst_exit_node;
    EthConnection eth_conn;

    bool operator==(const ExitNodeConnection& other) const {
        return src_exit_node == other.src_exit_node && dst_exit_node == other.dst_exit_node &&
               eth_conn == other.eth_conn;
    }
    bool operator<(const ExitNodeConnection& other) const {
        if (src_exit_node != other.src_exit_node) {
            return src_exit_node < other.src_exit_node;
        }
        if (dst_exit_node != other.dst_exit_node) {
            return dst_exit_node < other.dst_exit_node;
        }
        return eth_conn < other.eth_conn;
    }
};

using ExitNodeConnectionTable = std::unordered_map<std::string, std::vector<ExitNodeConnection>>;
using AsicConnectionEdge = std::pair<AsicID, std::vector<EthConnection>>;
using HostConnectionEdge = std::pair<std::string, std::vector<ExitNodeConnection>>;
using AsicTopology = std::unordered_map<AsicID, std::vector<AsicConnectionEdge>>;
using HostTopology = std::unordered_map<std::string, std::vector<HostConnectionEdge>>;

// Graph representing connectivity of ASICs (detailed representation) and
// Compute Nodes/Hosts (low resolution representation).
struct PhysicalConnectivityGraph {
    // Track the ASIC Topology per host.
    std::unordered_map<std::string, AsicTopology> asic_connectivity_graph;
    // Track the Host Topology across the distributed system.
    HostTopology host_connectivity_graph;
};

// Top-Level Global System Descriptor Data-Structure.

class PhysicalSystemDescriptor {
public:
    PhysicalSystemDescriptor(bool run_discovery = true);
    void run_discovery(bool run_global_discovery = true);
    // ASIC Topology Query APIs
    std::vector<AsicID> get_asic_neighbors(AsicID asic_id) const;
    std::vector<EthConnection> get_eth_connections(AsicID src_asic_id, AsicID dst_asic_id) const;
    const AsicTopology& get_asic_topology(const std::string& hostname) const;
    TrayID get_tray_id(AsicID asic_id) const;
    ASICLocation get_asic_location(AsicID asic_id) const;
    std::vector<AsicID> get_asics_connected_to_host(std::string hostname) const;

    // Host Topology Query APIs
    std::vector<std::string> get_host_neighbors(const std::string& hostname) const;
    std::vector<ExitNodeConnection> get_connecting_exit_nodes(
        const std::string& src_host, const std::string& dst_host) const;
    const HostTopology& get_host_topology() const;
    std::string get_host_name_for_asic(AsicID asic_id) const;
    UID get_u_id(const std::string& hostname);
    RackID get_rack_id(const std::string& hostname);
    AisleID get_aisle_id(const std::string& hostname);
    HallID get_hall_id(const std::string& hostname);
    std::vector<std::string> get_all_hostnames() const;
    std::string my_host_name() const;
    uint32_t get_rank_for_hostname(const std::string& host_name) const;

    // Generic Getters
    const PhysicalConnectivityGraph& get_system_graph() const { return system_graph_; }
    const std::unordered_map<AsicID, ASICDescriptor>& get_asic_descriptors() const { return asic_descriptors_; }
    const std::unordered_map<std::string, std::string>& get_host_mobo_name_map() const { return host_to_mobo_name_; }
    const std::unordered_map<std::string, uint32_t>& get_host_to_rank_map() const { return host_to_rank_; }
    const ExitNodeConnectionTable& get_exit_node_connection_table() const { return exit_node_connection_table_; }

    PhysicalConnectivityGraph& get_system_graph() { return system_graph_; }
    std::unordered_map<AsicID, ASICDescriptor>& get_asic_descriptors() { return asic_descriptors_; }
    std::unordered_map<std::string, std::string>& get_host_mobo_name_map() { return host_to_mobo_name_; }
    std::unordered_map<std::string, uint32_t>& get_host_to_rank_map() { return host_to_rank_; }
    ExitNodeConnectionTable& get_exit_node_connection_table() { return exit_node_connection_table_; }

    // Utility APIs to Print Physical System Descriptor
    void dump_to_yaml(const std::optional<std::string>& path_to_yaml = std::nullopt);
    void emit_to_text_proto(const std::optional<std::string>& path_to_text_proto = std::nullopt);

private:
    void run_local_discovery();
    void run_global_discovery();
    void clear();
    void merge(PhysicalSystemDescriptor&& other);
    void exchange_metadata(bool issue_gather);
    void generate_cross_host_connections();
    void remove_unresolved_nodes();
    void resolve_hostname_uniqueness();
    void validate_graphs();
    PhysicalConnectivityGraph system_graph_;
    std::unordered_map<AsicID, ASICDescriptor> asic_descriptors_;
    std::unordered_map<std::string, std::string> host_to_mobo_name_;
    std::unordered_map<std::string, uint32_t> host_to_rank_;
    ExitNodeConnectionTable exit_node_connection_table_;
    bool all_hostnames_unique_ = true;
};

}  // namespace tt::tt_metal
