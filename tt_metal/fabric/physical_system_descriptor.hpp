// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

namespace tt::tt_metal {

// TODO: These will be Strong Types
using asic_id_t = uint64_t;
using tray_id_t = uint32_t;
using n_id_t = uint32_t;
using rack_id_t = uint32_t;
using u_id_t = uint32_t;
using hall_id_t = uint32_t;
using aisle_id_t = uint32_t;

// Specifies an ethernet connection between 2 ASICs
// using channel ids on the src/dst ASICs
struct EthConnection {
    chan_id_t src_chan;
    chan_id_t dst_chan;
};

// Specifies an ethernet connection between 2 Compute
// Nodes using src/dst Exit Nodes (ASICs) and src/dst
// channel ids
struct ExitNodeConnection {
    asic_id_t src_exit_node;
    asic_id_t dst_exit_node;
    EthConnection eth_conn;
};

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
    PhysicalSystemDescriptor(bool perform_global_discovery = true);
    void run_discovery(bool perform_global_discovery = true);
    void validate_with_factory_desc(const std::string& path_to_factory_desc);
    void dump_to_yaml(conststd::string& path_to_yaml);

    // ASIC Topology Query APIs
    std::vector<asic_id_t> get_asic_neighbors(asic_id_t asic_id) const;
    std::vector<EthConnection> get_eth_connections(asic_id_t src_asic_id, asic_id_t dst_asic_id) const;
    const AsicTopology& get_asic_topology(const std::string& hostname) const;
    tray_id_t get_tray_id(asic_id_t asic_id) const;
    n_id_t get_n_id(n_id_t n_id) const;
    std::vector<asic_id_t> get_asics_connected_to_host(std::string hostname) const;

    // Query APIs for InterHost Connectivity
    std::vector<std::string> get_host_neighbors(const std::string& hostname) const;
    std::vector<ExitNodeConnection> get_connecting_exit_nodes(const std::string& src_host, const std::string& dst_host);
    const HostTopology& get_host_topology() const;
    std::string get_host_name_for_asic(asic_id_t asic_id) const;
    // Physical info that can be derived from the cabling spec (helps with clearer messages during physical validation).
    // At this point the FSD anc cabling spec have hostnames
    u_id_t get_u_id(const std::string& hostname);
    rack_id_t get_rack_id(const std::string& hostname);
    aisle_id_t get_aisle_id(const std::string& hostname);
    hall_id_t get_hall_id(const std::string& hostname);
    // Returns an ethernet path between two ASICs.
    // Maybe this is too advanced....
    // Maybe do get paths??
    std::vector<AsicConnectionEdge> get_path_between_asics(
        asic_id_t src_asic_id,
        asic_id_t dst_asic_id,
        bool terminate_at_exit_node,
        std::optional<uint32_t> chan_idx = std::nullopt) const;

private:
    PhysicalConnectivityGraph system_graph;
    YAML::Node serialized_desc;
};

}  // namespace tt::tt_metal
