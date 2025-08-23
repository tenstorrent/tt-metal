

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
#include "tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h"

namespace tt::tt_metal {

using asic_id_t = uint64_t;
using tray_id_t = uint32_t;
using n_id_t = uint32_t;
using rack_id_t = uint32_t;
using u_id_t = uint32_t;
using hall_id_t = uint32_t;
using aisle_id_t = uint32_t;

// Specifies an ethernet connection between 2 ASICs
// using channel ids on the src/dst ASICs

struct ASICDescriptor {
    tray_id_t tray_id;
    n_id_t n_id;
    BoardType board_type;
    asic_id_t unique_id;
    std::string host_name;
    // std::vector<chan_id_t>
};

struct EthConnection {
    tt_fabric::chan_id_t src_chan;
    tt_fabric::chan_id_t dst_chan;
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

struct ExitNodeConnection {
    asic_id_t src_exit_node;
    asic_id_t dst_exit_node;
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

// Can be used to specify the downstream ASICs a node in a graph of ASICs connects to.
// First entry in the pair specifies the Node/ASIC the current node is connected to.
// Second entry specifies the ethernet connections that build this edge (can have multiple
// ethernet channels connecting 2 ASICs)
using AsicConnectionEdge = std::pair<asic_id_t, std::vector<EthConnection>>;

// Can be used to specify the downstream Hosts a node in a graph of hosts connect to.
// First entry in the pair specifies the Node/Host the current node is connected to.
// Second entry specifies the exit node cinnections that build this edge (can have multiple
// exit nodes connecting 2 Compute Nodes).
using HostConnectionEdge = std::pair<std::string, std::vector<ExitNodeConnection>>;

// Graph of ASICs represented as an adjacency list (each entry in the list is an
// AsicConnectionEdge to a unique neighbor, each key in the map is a unique ASIC).
using AsicTopology = std::unordered_map<asic_id_t, std::vector<AsicConnectionEdge>>;

// Graph of Hosts represented as an adjacency list (each entry in the list is a
// HostConnectionEdge to a unique neighbor, each key in the map is a unique Host).
using HostTopology = std::unordered_map<std::string, std::vector<HostConnectionEdge>>;

// Top Level Data-Structure Representing connectivity of ASICs (detailed representation) and
// Compute Nodes/Hosts (low resolution representation).
struct PhysicalConnectivityGraph {
    // Tracks the ASIC Topology per host. Through this, the user can query
    // information specific to a local cluster.
    std::unordered_map<std::string, AsicTopology> asic_connectivity_graph;

    // Tracks the Host Topology across the distributed system. Through this,
    // the user can query how hosts are connected to each other.

    // By combining the asic_connectivity_graph and host_connectivity_graph,
    // a user can query how an connectivity information between ASICs mapped
    // to different hosts.
    HostTopology host_connectivity_graph;
};

// Main FSD Software Data-Structure, provoding:
// 1. Discovery APIs
// 2. Validation APIs
// 3. Graph Based Query APIs
// 4. System Query APIs
class PhysicalSystemDescriptor {
public:
    // Flexible discovery and validation. Allows users to:
    // 1. Perform global discovery in a single shot and then validate
    //    - Simple policy : Can report all errors at once.
    //    - Global discovery could be costly.
    // 2. Perform local discovery first, run validation and then run global discovery
    //    - Benefit: If lightweight local discovery fails, error out immediately instead
    //      of doing the global discovery.
    //    - Drawback: All errors can't be reported in a single shot.
    // 3. Update the physical representation at runtime by discovering the local or
    //    global cluster
    //    - This can be used by the provisioning SW to maintain an accurate representation
    //      of the system
    //    - Fairly expensive to repeatedly do at runtime for large systems
    PhysicalSystemDescriptor(bool perform_global_discovery = true, bool run_discovery = true);
    void run_discovery(bool perform_global_discovery = true);
    void validate_with_factory_desc(const std::string& path_to_factory_desc);
    void dump_to_yaml(const std::string& path_to_yaml);

    // ASIC Topology Query APIs
    std::vector<asic_id_t> get_asic_neighbors(asic_id_t asic_id) const;
    std::vector<EthConnection> get_eth_connections(asic_id_t src_asic_id, asic_id_t dst_asic_id) const;
    const AsicTopology& get_asic_topology(const std::string& hostname) const;
    tray_id_t get_tray_id(asic_id_t asic_id) const;
    n_id_t get_n_id(asic_id_t asic_id) const;
    std::vector<asic_id_t> get_asics_connected_to_host(std::string hostname) const;

    // Query APIs for InterHost Connectivity
    std::vector<std::string> get_host_neighbors(const std::string& hostname) const;
    std::vector<ExitNodeConnection> get_connecting_exit_nodes(
        const std::string& src_host, const std::string& dst_host) const;
    const HostTopology& get_host_topology() const;
    std::string get_host_name_for_asic(asic_id_t asic_id) const;
    // Physical info that can be derived from the cabling spec (helps with clearer messages during physical validation).
    // At this point the FSD anc cabling spec have hostnames
    u_id_t get_u_id(const std::string& hostname);
    rack_id_t get_rack_id(const std::string& hostname);
    aisle_id_t get_aisle_id(const std::string& hostname);
    hall_id_t get_hall_id(const std::string& hostname);

    const PhysicalConnectivityGraph& get_system_graph() const { return system_graph_; }
    const std::unordered_map<asic_id_t, ASICDescriptor>& get_asic_descriptors() const { return asic_descriptors_; }
    const std::unordered_map<std::string, std::string>& get_host_mobo_name_map() const { return host_to_mobo_name_; }
    const ExitNodeConnectionTable& get_exit_node_connection_table() const { return exit_node_connection_table_; }

    PhysicalConnectivityGraph& get_system_graph() { return system_graph_; }
    std::unordered_map<asic_id_t, ASICDescriptor>& get_asic_descriptors() { return asic_descriptors_; }
    std::unordered_map<std::string, std::string>& get_host_mobo_name_map() { return host_to_mobo_name_; }
    ExitNodeConnectionTable& get_exit_node_connection_table() { return exit_node_connection_table_; }

private:
    void run_local_discovery();
    void run_global_discovery();
    void merge(PhysicalSystemDescriptor&& other);
    void exchange_metadata(bool issue_gather);
    void generate_cross_host_connections();
    void remove_unresolved_nodes();
    PhysicalConnectivityGraph system_graph_;
    std::unordered_map<asic_id_t, ASICDescriptor> asic_descriptors_;
    std::unordered_map<std::string, std::string> host_to_mobo_name_;
    ExitNodeConnectionTable exit_node_connection_table_;
    // YAML::Node serialized_desc_;
};

}  // namespace tt::tt_metal
