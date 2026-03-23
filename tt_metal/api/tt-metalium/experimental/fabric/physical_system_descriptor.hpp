// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <umd/device/types/cluster_descriptor_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt_stl/strong_type.hpp>

namespace YAML {
class Node;
}

namespace tt {
enum class TargetDevice : std::uint8_t;
}

namespace tt::fabric::proto {
class PhysicalSystemDescriptor;
}  // namespace tt::fabric::proto

namespace tt::tt_metal {

// Forward declaration for discovery setter interface
namespace discovery_impl {
class DiscoverySetter;
}

// Live Ethernet Link Metrics
struct EthernetMetrics {
    uint32_t retrain_count = 0;
    uint32_t crc_error_count = 0;
    uint64_t corrected_codeword_count = 0;
    uint64_t uncorrected_codeword_count = 0;
};

using LocalEthernetMetrics = std::unordered_map<AsicID, std::unordered_map<uint8_t, EthernetMetrics>>;

// Specify Physical ASIC Attributes
struct ASICDescriptor {
    TrayID tray_id;
    ASICLocation asic_location;
    BoardType board_type = BoardType::UNKNOWN;
    AsicID unique_id;
    ChipId umd_unique_id;
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

}  // namespace tt::tt_metal

// Hash specialization for ExitNodeConnection that provides associative hashing
// The hash is the same regardless of src/dst order and eth_conn channel order
namespace std {
template <>
struct hash<tt::tt_metal::ExitNodeConnection> {
    std::size_t operator()(const tt::tt_metal::ExitNodeConnection& conn) const noexcept {
        // Get the underlying values from the StrongType wrappers
        uint64_t src_id = *conn.src_exit_node;
        uint64_t dst_id = *conn.dst_exit_node;

        // Sort the node IDs to make the hash associative
        uint64_t min_node = std::min(src_id, dst_id);
        uint64_t max_node = std::max(src_id, dst_id);

        // Sort the channel IDs to make the eth_conn hash associative
        uint8_t min_chan = std::min(conn.eth_conn.src_chan, conn.eth_conn.dst_chan);
        uint8_t max_chan = std::max(conn.eth_conn.src_chan, conn.eth_conn.dst_chan);

        // Use the same hash pattern as mesh_graph.hpp
        std::size_t seed = std::hash<uint64_t>{}(min_node);
        seed ^= std::hash<uint64_t>{}(max_node) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<uint8_t>{}(min_chan) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<uint8_t>{}(max_chan) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<bool>{}(conn.eth_conn.is_local) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};
}  // namespace std

namespace tt::tt_metal {

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
    // Minimal constructor - takes only target device type
    explicit PhysicalSystemDescriptor(tt::TargetDevice target_device_type);

    // Constructor generating a PhysicalSystemDescriptor based on a protobuf
    // descriptor file (can be used entirely offline).
    PhysicalSystemDescriptor(const std::string& mock_proto_desc_path);

    // Constructor generating a PhysicalSystemDescriptor based on a protobuf
    // descriptor (can be used entirely offline).
    PhysicalSystemDescriptor(const tt::fabric::proto::PhysicalSystemDescriptor& psd_proto);

    ~PhysicalSystemDescriptor();

    // Move constructor (move assignment not possible due to const reference member)
    PhysicalSystemDescriptor(PhysicalSystemDescriptor&&) = default;

    // Delete copy constructor, copy assignment, and move assignment operators
    PhysicalSystemDescriptor(const PhysicalSystemDescriptor&) = delete;
    PhysicalSystemDescriptor& operator=(const PhysicalSystemDescriptor&) = delete;
    PhysicalSystemDescriptor& operator=(PhysicalSystemDescriptor&&) = delete;

    // Public methods for re-discovery scenarios
    void clear();
    void merge(PhysicalSystemDescriptor&& other);

    // Discovery setter interface - allows discovery functions to set internal state
    // without requiring friend declarations with MPI types in public header
    void set_discovery_data(const std::string& local_hostname, uint32_t local_rank, bool all_hostnames_unique);

    // ASIC Topology Query APIs
    std::vector<AsicID> get_asic_neighbors(AsicID asic_id) const;
    std::vector<EthConnection> get_eth_connections(AsicID src_asic_id, AsicID dst_asic_id) const;
    const AsicTopology& get_asic_topology(const std::string& hostname) const;
    TrayID get_tray_id(AsicID asic_id) const;
    ASICLocation get_asic_location(AsicID asic_id) const;
    ChipId get_umd_unique_id(AsicID asic_id) const;
    std::vector<AsicID> get_asics_connected_to_host(const std::string& hostname) const;
    std::pair<AsicID, uint8_t> get_connected_asic_and_channel(AsicID asic_id, uint8_t chan_id) const;
    AsicID get_asic_id(const std::string& hostname, TrayID tray_id, ASICLocation asic_location) const;

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
    std::string get_hostname_for_rank(uint32_t rank) const;
    bool is_cross_host_eth_link(AsicID asic_id, uint8_t chan_id) const;

    // Generic Getters
    const PhysicalConnectivityGraph& get_system_graph() const { return system_graph_; }
    const std::unordered_map<AsicID, ASICDescriptor>& get_asic_descriptors() const { return asic_descriptors_; }
    const std::unordered_map<std::string, std::string>& get_host_mobo_name_map() const { return host_to_mobo_name_; }
    const std::unordered_map<std::string, uint32_t>& get_host_to_rank_map() const { return host_to_rank_; }
    const ExitNodeConnectionTable& get_exit_node_connection_table() const { return exit_node_connection_table_; }
    const tt::umd::semver_t& get_ethernet_firmware_version() const { return ethernet_firmware_version_; }
    const std::unordered_map<std::string, std::unordered_map<uint32_t, std::unordered_set<uint32_t>>>&
    get_pcie_devices_per_tray() const {
        return pcie_devices_per_tray_;
    }
    const std::unordered_map<std::string, std::unordered_map<uint32_t, ASICLocation>>& get_pcie_id_to_asic_location()
        const {
        return pcie_id_to_asic_location_;
    }

    tt::TargetDevice get_target_device_type() const { return target_device_type_; }
    bool get_all_hostnames_unique() const { return all_hostnames_unique_; }

    PhysicalConnectivityGraph& get_system_graph() { return system_graph_; }
    std::unordered_map<AsicID, ASICDescriptor>& get_asic_descriptors() { return asic_descriptors_; }
    std::unordered_map<std::string, std::string>& get_host_mobo_name_map() { return host_to_mobo_name_; }
    std::unordered_map<std::string, uint32_t>& get_host_to_rank_map() { return host_to_rank_; }
    ExitNodeConnectionTable& get_exit_node_connection_table() { return exit_node_connection_table_; }
    tt::umd::semver_t& get_ethernet_firmware_version() { return ethernet_firmware_version_; }
    std::unordered_map<std::string, std::unordered_map<uint32_t, std::unordered_set<uint32_t>>>&
    get_pcie_devices_per_tray() {
        return pcie_devices_per_tray_;
    }
    std::unordered_map<std::string, std::unordered_map<uint32_t, ASICLocation>>& get_pcie_id_to_asic_location() {
        return pcie_id_to_asic_location_;
    }

    // Utility APIs to Print Physical System Descriptor
    void dump_to_yaml(const std::optional<std::string>& path_to_yaml = std::nullopt) const;
    YAML::Node generate_yaml_node() const;
    void emit_to_text_proto(const std::optional<std::string>& file_path = std::nullopt) const;

private:
    tt::TargetDevice target_device_type_;
    PhysicalConnectivityGraph system_graph_;
    std::unordered_map<AsicID, ASICDescriptor> asic_descriptors_;
    std::unordered_map<std::string, std::string> host_to_mobo_name_;
    std::unordered_map<std::string, uint32_t> host_to_rank_;
    ExitNodeConnectionTable exit_node_connection_table_;
    bool all_hostnames_unique_ = true;
    tt::umd::semver_t ethernet_firmware_version_;
    std::unordered_map<std::string, std::unordered_map<uint32_t, std::unordered_set<uint32_t>>> pcie_devices_per_tray_;
    std::unordered_map<std::string, std::unordered_map<uint32_t, ASICLocation>> pcie_id_to_asic_location_;

    // Local hostname and rank set by discovery (for my_host_name())
    std::string local_hostname_;
    uint32_t local_rank_ = 0;
};

}  // namespace tt::tt_metal
