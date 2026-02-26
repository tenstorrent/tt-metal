// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <set>
#include <fstream>
#include <climits>

#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include "tt_metal/fabric/serialization/physical_system_descriptor_serialization.hpp"
#include <tt_stl/assert.hpp>
#include "llrt/tt_target_device.hpp"
#include <unistd.h>

namespace tt::tt_metal {

/**************************************************************************************************
 Discovery helper functions
**************************************************************************************************/

namespace {

std::string get_host_name() {
    char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, sizeof(hostname));
    return std::string(hostname);
}

struct EthEndpoint {
    AsicID board_id;
    uint8_t chan_id;

    auto operator<=>(const EthEndpoint&) const = default;
};

}  // namespace

/**************************************************************************************************
 PhysicalSystemDescriptor implementation
**************************************************************************************************/

PhysicalSystemDescriptor::PhysicalSystemDescriptor(tt::TargetDevice target_device_type) :
    target_device_type_(target_device_type) {}

PhysicalSystemDescriptor::PhysicalSystemDescriptor(const std::string& mock_proto_desc_path) :
    // Deserialize the proto descriptor and move all its members directly
    // This avoids discovery and merge operations for mock proto descriptors
    PhysicalSystemDescriptor proto_desc =
        deserialize_physical_system_descriptor_from_text_proto_file(mock_proto_desc_path);

    // Move all members directly from the deserialized descriptor using non-const getters
    target_device_type_ = proto_desc.get_target_device_type();
    system_graph_ = std::move(proto_desc.get_system_graph());
    asic_descriptors_ = std::move(proto_desc.get_asic_descriptors());
    host_to_mobo_name_ = std::move(proto_desc.get_host_mobo_name_map());
    host_to_rank_ = std::move(proto_desc.get_host_to_rank_map());
    exit_node_connection_table_ = std::move(proto_desc.get_exit_node_connection_table());
    ethernet_firmware_version_ = proto_desc.get_ethernet_firmware_version();
    pcie_devices_per_tray_ = std::move(proto_desc.get_pcie_devices_per_tray());
    pcie_id_to_asic_location_ = std::move(proto_desc.get_pcie_id_to_asic_location());
}

PhysicalSystemDescriptor::~PhysicalSystemDescriptor() = default;

void PhysicalSystemDescriptor::set_discovery_data(
    const std::string& local_hostname, uint32_t local_rank, bool all_hostnames_unique) {
    local_hostname_ = local_hostname;
    local_rank_ = local_rank;
    all_hostnames_unique_ = all_hostnames_unique;
}

void PhysicalSystemDescriptor::clear() {
    // Erase all contents in all data structures
    system_graph_.asic_connectivity_graph.clear();
    system_graph_.host_connectivity_graph.clear();
    asic_descriptors_.clear();
    host_to_mobo_name_.clear();
    host_to_rank_.clear();
    exit_node_connection_table_.clear();
    pcie_devices_per_tray_.clear();
    pcie_id_to_asic_location_.clear();
    local_hostname_.clear();
    local_rank_ = 0;
    all_hostnames_unique_ = true;
    ethernet_firmware_version_ = tt::umd::semver_t(0, 0, 0);
}

void PhysicalSystemDescriptor::merge(PhysicalSystemDescriptor&& other) {
    for (auto& [host_name, asic_graph] : other.system_graph_.asic_connectivity_graph) {
        system_graph_.asic_connectivity_graph[host_name] = std::move(asic_graph);
    }
    for (auto& [host_name, host_connectivity] : other.system_graph_.host_connectivity_graph) {
        system_graph_.host_connectivity_graph[host_name] = std::move(host_connectivity);
    }
    for (auto& [asic_id, asic_desc] : other.get_asic_descriptors()) {
        asic_descriptors_[asic_id] = std::move(asic_desc);
    }
    for (auto& [host_name, mobo_name] : other.get_host_mobo_name_map()) {
        host_to_mobo_name_[host_name] = std::move(mobo_name);
    }
    for (auto& [host_name, rank] : other.get_host_to_rank_map()) {
        host_to_rank_[host_name] = rank;
    }
    for (auto& [host_name, exit_connections] : other.exit_node_connection_table_) {
        exit_node_connection_table_[host_name] = std::move(exit_connections);
    }
    for (auto& [host_name, tray_map] : other.get_pcie_devices_per_tray()) {
        pcie_devices_per_tray_[host_name] = std::move(tray_map);
    }
    for (auto& [host_name, pcie_map] : other.get_pcie_id_to_asic_location()) {
        pcie_id_to_asic_location_[host_name] = std::move(pcie_map);
    }

    // Preserve discovery identity and firmware version from source. Required for clear()+merge()
    // re-discovery flow: caller clears destination PSD, then merges a newly discovered PSD.
    // Without this, my_host_name() would fall back incorrectly and ethernet_firmware_version_
    // would remain at cleared default (0.0.0).
    if (!other.local_hostname_.empty()) {
        local_hostname_ = std::move(other.local_hostname_);
        local_rank_ = other.local_rank_;
        all_hostnames_unique_ = other.all_hostnames_unique_;
    }
    ethernet_firmware_version_ = other.ethernet_firmware_version_;

    // Merging PhysicalSystemDescriptors using mock and real clusters is undefined and unsupported
    TT_FATAL(
        target_device_type_ == other.target_device_type_,
        "Cannot merge physical and mock/simulation cluster physical system descriptors.");
}

YAML::Node PhysicalSystemDescriptor::generate_yaml_node() const {
    YAML::Node root;
    YAML::Node compute_nodes;
    YAML::Node local_eth_connections(YAML::NodeType::Sequence);
    YAML::Node global_eth_connections(YAML::NodeType::Sequence);

    std::set<std::pair<EthEndpoint, EthEndpoint>> processed_connections;
    for (const auto& [host_name, mobo_name] : host_to_mobo_name_) {
        YAML::Node host_node;
        YAML::Node tray_groups;
        host_node["motherboard"] = mobo_name;

        std::map<TrayID, std::vector<ASICDescriptor>> grouped_asics;

        for (const auto& asic : system_graph_.asic_connectivity_graph.at(host_name)) {
            AsicID asic_id = asic.first;
            TrayID tray_id = asic_descriptors_.at(asic_id).tray_id;
            grouped_asics[tray_id].push_back(asic_descriptors_.at(asic_id));
        }

        for (const auto& group : grouped_asics) {
            YAML::Node tray_group;
            tray_group["tray_id"] = *(group.first);  // tray_id
            tray_group["board_type"] = enchantum::to_string(group.second.front().board_type);
            std::vector<ASICDescriptor> sorted_asics = group.second;
            std::sort(sorted_asics.begin(), sorted_asics.end(), [](const ASICDescriptor& a, const ASICDescriptor& b) {
                return a.asic_location < b.asic_location;
            });
            // Create asics array
            YAML::Node asics_array;
            for (const auto& asic : sorted_asics) {
                YAML::Node asic_node;
                asic_node["asic_location"] = *(asic.asic_location);
                asic_node["asic_id"] = *(asic.unique_id);
                asics_array.push_back(asic_node);
            }
            tray_group["asics"] = asics_array;
            tray_groups.push_back(tray_group);
        }
        host_node["asic_info"] = tray_groups;
        compute_nodes[host_name] = host_node;

        for (const auto& asic : system_graph_.asic_connectivity_graph.at(host_name)) {
            auto src_asic_id = asic.first;
            const auto& src_asic_desc = asic_descriptors_.at(src_asic_id);
            for (const auto& edge : asic.second) {
                auto dst_asic_id = edge.first;
                const auto& dst_asic_desc = asic_descriptors_.at(dst_asic_id);
                for (const auto& eth_conn : edge.second) {
                    EthEndpoint src_id{src_asic_id, eth_conn.src_chan};
                    EthEndpoint dst_id{dst_asic_id, eth_conn.dst_chan};
                    auto connection_key = std::make_pair(std::min(src_id, dst_id), std::max(src_id, dst_id));

                    if (processed_connections.contains(connection_key)) {
                        continue;
                    }
                    processed_connections.insert(connection_key);

                    YAML::Node src_conn_node;
                    YAML::Node dst_conn_node;
                    YAML::Node connection_pair(YAML::NodeType::Sequence);
                    connection_pair.SetStyle(YAML::EmitterStyle::Flow);
                    src_conn_node["host_name"] = src_asic_desc.host_name;
                    dst_conn_node["host_name"] = dst_asic_desc.host_name;
                    src_conn_node["tray_id"] = *(src_asic_desc.tray_id);
                    src_conn_node["asic_location"] = *(src_asic_desc.asic_location);
                    dst_conn_node["tray_id"] = *(dst_asic_desc.tray_id);
                    dst_conn_node["asic_location"] = *(dst_asic_desc.asic_location);
                    src_conn_node["chan_id"] = +eth_conn.src_chan;
                    dst_conn_node["chan_id"] = +eth_conn.dst_chan;

                    connection_pair.push_back(src_conn_node);
                    connection_pair.push_back(dst_conn_node);

                    if (eth_conn.is_local) {
                        local_eth_connections.push_back(connection_pair);
                    } else {
                        global_eth_connections.push_back(connection_pair);
                    }
                }
            }
        }
    }

    root["compute_node_specs"] = compute_nodes;
    root["local_eth_connections"] = local_eth_connections;
    root["global_eth_connections"] = global_eth_connections;

    return root;
}

void PhysicalSystemDescriptor::dump_to_yaml(const std::optional<std::string>& path_to_yaml) const {
    YAML::Node root = generate_yaml_node();

    if (path_to_yaml.has_value()) {
        std::ofstream fout(path_to_yaml.value());
        if (!fout.is_open()) {
            TT_THROW("Failed to open file for writing: {}", path_to_yaml.value());
        }
        fout << root;
        if (fout.fail()) {
            TT_THROW("Failed to write YAML content to file: {}", path_to_yaml.value());
        }
    } else {
        std::cout << root << std::endl;
    }
}

void PhysicalSystemDescriptor::emit_to_text_proto(const std::optional<std::string>& file_path) const {
    emit_physical_system_descriptor_to_text_proto(*this, file_path);
}

std::vector<AsicID> PhysicalSystemDescriptor::get_asic_neighbors(AsicID asic_id) const {
    for (const auto& [host, asic_group] : system_graph_.asic_connectivity_graph) {
        if (asic_group.contains(asic_id)) {
            std::vector<AsicID> neighbors;
            for (const auto& edge : asic_group.at(asic_id)) {
                neighbors.push_back(edge.first);
            }
            return neighbors;
        }
    }
    return {};
}

std::vector<EthConnection> PhysicalSystemDescriptor::get_eth_connections(AsicID src_asic, AsicID dst_asic) const {
    for (const auto& [host, asic_group] : system_graph_.asic_connectivity_graph) {
        if (asic_group.contains(src_asic)) {
            for (const auto& edge : asic_group.at(src_asic)) {
                if (edge.first == dst_asic) {
                    return edge.second;
                }
            }
        }
    }
    return {};
}

const AsicTopology& PhysicalSystemDescriptor::get_asic_topology(const std::string& hostname) const {
    TT_FATAL(system_graph_.asic_connectivity_graph.contains(hostname), "No ASIC topology found for host {}", hostname);
    return system_graph_.asic_connectivity_graph.at(hostname);
}

TrayID PhysicalSystemDescriptor::get_tray_id(AsicID asic_id) const {
    TT_FATAL(asic_descriptors_.contains(asic_id), "No ASIC descriptor found for asic_id {}", asic_id);
    return asic_descriptors_.at(asic_id).tray_id;
}

ASICLocation PhysicalSystemDescriptor::get_asic_location(AsicID asic_id) const {
    TT_FATAL(asic_descriptors_.contains(asic_id), "No ASIC descriptor found for asic_id {}", asic_id);
    return asic_descriptors_.at(asic_id).asic_location;
}

std::vector<AsicID> PhysicalSystemDescriptor::get_asics_connected_to_host(const std::string& hostname) const {
    std::vector<AsicID> asics;
    if (system_graph_.asic_connectivity_graph.contains(hostname)) {
        for (const auto& [asic_id, _] : system_graph_.asic_connectivity_graph.at(hostname)) {
            asics.push_back(asic_id);
        }
    }
    return asics;
}

bool PhysicalSystemDescriptor::is_cross_host_eth_link(AsicID asic_id, uint8_t chan_id) const {
    for (const auto& [host, asic_group] : system_graph_.asic_connectivity_graph) {
        if (this->get_host_name_for_asic(asic_id) != host) {
            continue;
        }
        const auto& connections = asic_group.at(asic_id);
        auto connection_it = std::find_if(connections.begin(), connections.end(), [&](const auto& connection) {
            // Check if this chan_id is a src_chan in any of the eth_connections
            return std::find_if(connection.second.begin(), connection.second.end(), [&](const auto& eth_conn) {
                       return eth_conn.src_chan == chan_id;
                   }) != connection.second.end();
        });
        TT_FATAL(
            connection_it != connections.end(),
            "Channel {} not found in asic connectivity graph for asic {}",
            chan_id,
            asic_id);
        auto connected_asic = connection_it->first;
        return this->get_host_name_for_asic(connected_asic) != host;
    }
    TT_THROW("Asic {} not found in any host's asic connectivity graph", asic_id);
    return false;
}

std::vector<std::string> PhysicalSystemDescriptor::get_host_neighbors(const std::string& hostname) const {
    TT_FATAL(
        system_graph_.host_connectivity_graph.contains(hostname), "No Host connectivity found for host {}", hostname);
    std::vector<std::string> neighbors;
    for (const auto& edge : system_graph_.host_connectivity_graph.at(hostname)) {
        neighbors.push_back(edge.first);
    }
    return neighbors;
}

std::vector<ExitNodeConnection> PhysicalSystemDescriptor::get_connecting_exit_nodes(
    const std::string& src_host, const std::string& dst_host) const {
    TT_FATAL(
        system_graph_.host_connectivity_graph.contains(src_host), "No Host connectivity found for host {}", src_host);
    for (const auto& edge : system_graph_.host_connectivity_graph.at(src_host)) {
        if (edge.first == dst_host) {
            return edge.second;
        }
    }
    return {};
}

std::pair<AsicID, uint8_t> PhysicalSystemDescriptor::get_connected_asic_and_channel(
    AsicID asic_id, uint8_t chan_id) const {
    auto host = asic_descriptors_.at(asic_id).host_name;
    auto asic_graph = system_graph_.asic_connectivity_graph.at(host);
    for (const auto& [src_asic, edges] : asic_graph) {
        if (src_asic != asic_id) {
            continue;
        }
        for (const auto& edge : edges) {
            auto dst_asic = edge.first;

            for (const auto& eth_conn : edge.second) {
                if (eth_conn.src_chan == chan_id) {
                    return {dst_asic, eth_conn.dst_chan};
                }
            }
        }
    }
    TT_THROW("No connected ASIC and channel found for asic ID {} and channel ID {}", asic_id, chan_id);
    return {AsicID{0}, 0};
}

AsicID PhysicalSystemDescriptor::get_asic_id(
    const std::string& hostname, TrayID tray_id, ASICLocation asic_location) const {
    for (const auto& [asic_id, asic_descriptor] : asic_descriptors_) {
        if (asic_descriptor.host_name == hostname && asic_descriptor.tray_id == tray_id &&
            asic_descriptor.asic_location == asic_location) {
            return asic_id;
        }
    }
    TT_THROW("No ASIC ID found at hostname {}, tray ID {}, and ASIC location {}", hostname, *tray_id, *asic_location);
    return AsicID{0};
}

const HostTopology& PhysicalSystemDescriptor::get_host_topology() const {
    return system_graph_.host_connectivity_graph;
}

std::vector<std::string> PhysicalSystemDescriptor::get_all_hostnames() const {
    std::vector<std::string> hostnames;
    hostnames.reserve(system_graph_.asic_connectivity_graph.size());
    for (const auto& [host, _] : system_graph_.asic_connectivity_graph) {
        hostnames.push_back(host);
    }
    return hostnames;
}

std::string PhysicalSystemDescriptor::my_host_name() const {
    if (!local_hostname_.empty()) {
        // Discovery has set local_hostname_ and local_rank_
        if (all_hostnames_unique_) {
            return local_hostname_;
        }
        return local_hostname_ + "_" + std::to_string(local_rank_);
    }
    // Fallback for file-based PSD (no discovery) - assume hostnames are unique
    return get_host_name();
}

uint32_t PhysicalSystemDescriptor::get_rank_for_hostname(const std::string& host_name) const {
    TT_FATAL(host_to_rank_.contains(host_name), "Rank for host {} not found", host_name);
    return host_to_rank_.at(host_name);
}

std::string PhysicalSystemDescriptor::get_hostname_for_rank(uint32_t rank) const {
    for (const auto& [host, host_rank] : host_to_rank_) {
        if (host_rank == rank) {
            return host;
        }
    }
    TT_THROW("Hostname for rank {} not found", rank);
}

std::string PhysicalSystemDescriptor::get_host_name_for_asic(AsicID asic_id) const {
    TT_FATAL(asic_descriptors_.contains(asic_id), "No ASIC descriptor found for asic_id {}", asic_id);
    return asic_descriptors_.at(asic_id).host_name;
}

UID PhysicalSystemDescriptor::get_u_id(const std::string& /*hostname*/) {
    TT_THROW("Querying Host UID requires the Cable Spec which is not currently supported.");
}

RackID PhysicalSystemDescriptor::get_rack_id(const std::string& /*hostname*/) {
    TT_THROW("Querying Host Rack ID requires the Cable Spec which is not currently supported.");
}

AisleID PhysicalSystemDescriptor::get_aisle_id(const std::string& /*hostname*/) {
    TT_THROW("Querying Host Aisle ID requires the Cable Spec which is not currently supported.");
}

HallID PhysicalSystemDescriptor::get_hall_id(const std::string& /*hostname*/) {
    TT_THROW("Querying Host Hall ID requires the Cable Spec which is not currently supported.");
}

}  // namespace tt::tt_metal
