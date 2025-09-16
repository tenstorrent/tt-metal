// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <yaml-cpp/yaml.h>

#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/distributed_context.hpp>
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/fabric/serialization/physical_system_descriptor_serialization.hpp"

namespace tt::tt_metal {

namespace {

std::string get_host_name() {
    char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, sizeof(hostname));
    return std::string(hostname);
}

std::string get_mobo_name() {
    std::ifstream file("/sys/class/dmi/id/board_name");
    std::string motherboard;

    if (file.is_open()) {
        std::getline(file, motherboard);
        file.close();
    }

    return motherboard;
}

TrayID get_tray_id_for_chip(chip_id_t chip_id, const std::string& mobo_name) {
    static const std::unordered_map<std::string, std::vector<uint16_t>> mobo_to_bus_ids = {
        {"SIENAD8-2L2T", {0xc1, 0x01, 0x41, 0x42}},
        {"X12DPG-QT6", {0xb1, 0xca, 0x31, 0x4b}},
    };

    if (mobo_to_bus_ids.find(mobo_name) == mobo_to_bus_ids.end()) {
        return TrayID{0};
    }
    const auto& ordered_bus_ids = mobo_to_bus_ids.at(mobo_name);
    auto bus_id = tt::tt_metal::MetalContext::instance().get_cluster().get_bus_id(chip_id);
    auto bus_id_it = std::find(ordered_bus_ids.begin(), ordered_bus_ids.end(), bus_id);
    TT_FATAL(bus_id_it != ordered_bus_ids.end(), "Bus ID {} not found.", bus_id);
    auto tray_id = std::distance(ordered_bus_ids.begin(), bus_id_it) + 1;
    return TrayID{tray_id};
}

std::pair<TrayID, ASICLocation> get_asic_position(chip_id_t chip_id) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto cluster_desc = cluster.get_cluster_desc();
    if (cluster_desc->get_board_type(chip_id) == BoardType::UBB) {
        constexpr std::string_view ubb_mobo_name = "S7T-MB";

        TT_FATAL(get_mobo_name() == ubb_mobo_name, "UBB systems must use S7T-MB motherboard.");
        auto ubb_id = tt::tt_fabric::get_ubb_id(chip_id);
        return {TrayID{ubb_id.tray_id}, ASICLocation{ubb_id.asic_id}};
    } else {
        auto tray_id = get_tray_id_for_chip(chip_id, get_mobo_name());
        ASICLocation asic_location;
        if (cluster.arch() == tt::ARCH::WORMHOLE_B0) {
            // Derive ASIC Location based on the tunnel depth for Wormhole systems
            // TODO: Remove this once UMD populates the ASIC Location for WH systems.
            auto mmio_device = cluster.get_associated_mmio_device(chip_id);
            auto tunnels = cluster.get_tunnels_from_mmio_device(mmio_device);
            for (auto tunnel = 0; tunnel < tunnels.size(); tunnel++) {
                const auto& devices_on_tunnel = tunnels[tunnel];
                auto device_it = std::find(devices_on_tunnel.begin(), devices_on_tunnel.end(), chip_id);
                if (device_it != devices_on_tunnel.end()) {
                    asic_location = ASICLocation{device_it - devices_on_tunnel.begin()};
                    break;
                }
            }
        } else if (cluster.arch() == tt::ARCH::BLACKHOLE) {
            // Query ASIC Location from the Cluster Descriptor for BH.
            asic_location = ASICLocation{cluster_desc->get_asic_location(chip_id)};
        } else {
            TT_THROW("Unrecognized Architecture. Cannot determine asic location.");
        }
        return {tray_id, asic_location};
    }
}

struct EthEndpoint {
    AsicID board_id;
    uint8_t chan_id;

    auto operator<=>(const EthEndpoint&) const = default;
};

}  // namespace

PhysicalSystemDescriptor::PhysicalSystemDescriptor(bool run_discovery) {
    if (run_discovery) {
        this->run_discovery();
    }
}

void PhysicalSystemDescriptor::resolve_hostname_uniqueness() {
    using namespace tt::tt_metal::distributed::multihost;
    constexpr uint32_t controller_rank = 0;
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    auto my_rank = *(distributed_context.rank());

    if (my_rank == controller_rank) {
        std::vector<std::string> hostnames = {};
        hostnames.push_back(get_host_name());
        for (std::size_t rank = 0; rank < *(distributed_context.size()); rank++) {
            if (rank != controller_rank) {
                std::size_t peer_hostname_size = 0;
                distributed_context.recv(
                    tt::stl::Span<std::byte>(
                        reinterpret_cast<std::byte*>(&peer_hostname_size), sizeof(peer_hostname_size)),
                    Rank{rank},
                    Tag{0});
                std::vector<uint8_t> serialized_peer_hostname(peer_hostname_size);
                distributed_context.recv(
                    tt::stl::as_writable_bytes(
                        tt::stl::Span<uint8_t>(serialized_peer_hostname.data(), serialized_peer_hostname.size())),
                    Rank{rank},
                    Tag{0});

                hostnames.push_back(std::string(serialized_peer_hostname.begin(), serialized_peer_hostname.end()));
            }
        }
        all_hostnames_unique_ = std::set<std::string>(hostnames.begin(), hostnames.end()).size() == hostnames.size();

        for (std::size_t rank = 0; rank < *(distributed_context.size()); rank++) {
            if (rank != controller_rank) {
                distributed_context.send(
                    tt::stl::Span<std::byte>(
                        reinterpret_cast<std::byte*>(&all_hostnames_unique_), sizeof(all_hostnames_unique_)),
                    Rank{rank},
                    Tag{0});
            }
        }
    } else {
        auto host_name = get_host_name();
        auto serialized_hostname = std::vector<uint8_t>(host_name.begin(), host_name.end());
        std::size_t serialized_hostname_size = serialized_hostname.size();
        distributed_context.send(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&serialized_hostname_size), sizeof(serialized_hostname_size)),
            Rank{controller_rank},
            Tag{0});
        distributed_context.send(
            tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(serialized_hostname.data(), serialized_hostname.size())),
            Rank{controller_rank},
            Tag{0});

        distributed_context.recv(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(&all_hostnames_unique_), sizeof(all_hostnames_unique_)),
            Rank{controller_rank},
            Tag{0});
    }
}

void PhysicalSystemDescriptor::run_discovery(bool run_global_discovery) {
    this->resolve_hostname_uniqueness();
    this->run_local_discovery();
    if (run_global_discovery) {
        this->run_global_discovery();
    }
}

void PhysicalSystemDescriptor::clear() {
    // Erase all contents in all data structures
    system_graph_.asic_connectivity_graph.clear();
    system_graph_.host_connectivity_graph.clear();
    asic_descriptors_.clear();
    host_to_mobo_name_.clear();
    host_to_rank_.clear();
    exit_node_connection_table_.clear();
}

void PhysicalSystemDescriptor::run_local_discovery() {
    this->clear();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    const auto& chip_unique_ids = cluster.get_unique_chip_ids();
    const auto& eth_connections = cluster.get_ethernet_connections();
    auto cross_host_eth_connections = cluster.get_ethernet_connections_to_remote_devices();
    auto cluster_desc = cluster.get_cluster_desc();

    auto my_rank = *(distributed_context.rank());
    auto hostname = this->my_host_name();
    host_to_mobo_name_[hostname] = get_mobo_name();
    host_to_rank_[hostname] = my_rank;

    auto& asic_graph = system_graph_.asic_connectivity_graph[hostname];
    auto& exit_nodes = exit_node_connection_table_[hostname];

    for (const auto& [src, conn] : eth_connections) {
        auto src_unique_id = AsicID{chip_unique_ids.at(src)};
        // Populate ASIC Descriptor with Physical Information
        auto [tray_id, asic_location] = get_asic_position(src);
        asic_descriptors_[src_unique_id] =
            ASICDescriptor{TrayID{tray_id}, asic_location, cluster_desc->get_board_type(src), src_unique_id, hostname};

        std::unordered_map<chip_id_t, size_t> visited_dst;
        // Populate ASIC Graph for Current Host
        for (auto& [chan, dst] : conn) {
            auto dst_chip = std::get<0>(dst);
            auto dst_chan = std::get<1>(dst);
            if (visited_dst.find(dst_chip) == visited_dst.end()) {
                // This neighbor has not been visited. Add it to the graph and mark visited.
                asic_graph[src_unique_id].push_back(
                    {AsicID{chip_unique_ids.at(dst_chip)}, {EthConnection(chan, dst_chan, true)}});
                visited_dst[dst_chip] = asic_graph[src_unique_id].size() - 1;
            } else {
                // This neighbor has already been visited. There is more than one channel to it.
                // Update the existing entry with the new channel.
                asic_graph[src_unique_id][visited_dst[dst_chip]].second.push_back(EthConnection(chan, dst_chan, true));
            }
        }
    }

    for (const auto& [local_chip_id, eth_link_info] : cross_host_eth_connections) {
        auto local_unique_id = AsicID{chip_unique_ids.at(local_chip_id)};
        std::unordered_map<AsicID, size_t> visited_dst;
        for (const auto& [eth_chan, remote_info] : eth_link_info) {
            auto dst_unique_id = AsicID{std::get<0>(remote_info)};
            auto dst_chan = std::get<1>(remote_info);
            if (visited_dst.find(dst_unique_id) == visited_dst.end()) {
                asic_graph[local_unique_id].push_back({dst_unique_id, {EthConnection(eth_chan, dst_chan, false)}});
                visited_dst[dst_unique_id] = asic_graph[local_unique_id].size() - 1;
            } else {
                asic_graph[local_unique_id][visited_dst[dst_unique_id]].second.push_back(
                    EthConnection(eth_chan, dst_chan, false));
            }
            exit_nodes.push_back(ExitNodeConnection{
                .src_exit_node = local_unique_id,
                .dst_exit_node = dst_unique_id,
                .eth_conn = EthConnection(eth_chan, dst_chan, false)});
        }
    }
    system_graph_.host_connectivity_graph[hostname] = {};
}

void PhysicalSystemDescriptor::run_global_discovery() {
    using namespace tt::tt_metal::distributed::multihost;
    constexpr uint32_t controller_rank = 0;
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    auto my_rank = *(distributed_context.rank());
    this->exchange_metadata(true);
    if (my_rank == controller_rank) {
        this->remove_unresolved_nodes();
        this->generate_cross_host_connections();
        this->validate_graphs();
    }
    this->exchange_metadata(false);
    distributed_context.barrier();
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
}

void PhysicalSystemDescriptor::remove_unresolved_nodes() {
    for (auto& [host, asic_group] : system_graph_.asic_connectivity_graph) {
        for (auto& [src_asic, edges] : asic_group) {
            std::erase_if(
                edges, [&](const auto& pair) { return asic_descriptors_.find(pair.first) == asic_descriptors_.end(); });
        }
    }

    for (auto& [host, exit_nodes] : exit_node_connection_table_) {
        std::erase_if(exit_nodes, [&](const auto& exit_node) {
            return asic_descriptors_.find(exit_node.src_exit_node) == asic_descriptors_.end() ||
                   asic_descriptors_.find(exit_node.dst_exit_node) == asic_descriptors_.end();
        });
    }
}

void PhysicalSystemDescriptor::exchange_metadata(bool issue_gather) {
    using namespace tt::tt_metal::distributed::multihost;
    constexpr uint32_t controller_rank = 0;
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    auto my_rank = *(distributed_context.rank());
    std::set<uint32_t> sender_ranks;
    std::set<uint32_t> receiver_ranks;

    if (issue_gather) {
        receiver_ranks.insert(controller_rank);
        for (std::size_t rank = 0; rank < *(distributed_context.size()); rank++) {
            if (rank != controller_rank) {
                sender_ranks.insert(rank);
            }
        }
    } else {
        sender_ranks.insert(controller_rank);
        for (std::size_t rank = 0; rank < *(distributed_context.size()); rank++) {
            if (rank != controller_rank) {
                receiver_ranks.insert(rank);
            }
        }
    }

    if (sender_ranks.find(my_rank) != sender_ranks.end()) {
        auto serialized_desc = serialize_physical_system_descriptor_to_bytes(*this);
        std::size_t desc_size = serialized_desc.size();

        for (auto rank : receiver_ranks) {
            distributed_context.send(
                tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&desc_size), sizeof(desc_size)),
                Rank{rank},
                Tag{0});

            distributed_context.send(
                tt::stl::as_writable_bytes(tt::stl::Span<uint8_t>(serialized_desc.data(), serialized_desc.size())),
                Rank{rank},
                Tag{0});
        }
    } else {
        for (auto rank : sender_ranks) {
            std::size_t peer_descriptor_size = 0;
            distributed_context.recv(
                tt::stl::Span<std::byte>(
                    reinterpret_cast<std::byte*>(&peer_descriptor_size), sizeof(peer_descriptor_size)),
                Rank{rank},
                Tag{0});
            std::vector<uint8_t> serialized_peer_desc(peer_descriptor_size);
            distributed_context.recv(
                tt::stl::as_writable_bytes(
                    tt::stl::Span<uint8_t>(serialized_peer_desc.data(), serialized_peer_desc.size())),
                Rank{rank},
                Tag{0});
            auto peer_desc = deserialize_physical_system_descriptor_from_bytes(serialized_peer_desc);
            this->merge(std::move(peer_desc));
        }
    }
    distributed_context.barrier();
}

void PhysicalSystemDescriptor::generate_cross_host_connections() {
    for (const auto& [host, exit_nodes] : exit_node_connection_table_) {
        std::unordered_map<std::string, size_t> visited_hosts;
        for (const auto& [candidate_host, candidate_exit_nodes] : exit_node_connection_table_) {
            if (host == candidate_host) {
                continue;  // Skip self connections
            }
            for (const auto& exit_node : exit_nodes) {
                for (const auto& candidate_node : candidate_exit_nodes) {
                    if (exit_node.src_exit_node == candidate_node.dst_exit_node &&
                        candidate_node.src_exit_node == exit_node.dst_exit_node &&
                        exit_node.eth_conn.src_chan == candidate_node.eth_conn.dst_chan &&
                        exit_node.eth_conn.dst_chan == candidate_node.eth_conn.src_chan) {
                        if (visited_hosts.find(candidate_host) == visited_hosts.end()) {
                            system_graph_.host_connectivity_graph[host].push_back({candidate_host, {exit_node}});
                            visited_hosts[candidate_host] = system_graph_.host_connectivity_graph[host].size() - 1;
                        } else {
                            system_graph_.host_connectivity_graph[host][visited_hosts[candidate_host]].second.push_back(
                                exit_node);
                        }
                        break;
                    }
                }
            }
        }
    }
}

void PhysicalSystemDescriptor::dump_to_yaml(const std::optional<std::string>& path_to_yaml) {
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

        for (const auto& asic : system_graph_.asic_connectivity_graph[host_name]) {
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

        for (const auto& asic : system_graph_.asic_connectivity_graph[host_name]) {
            auto src_asic_id = asic.first;
            const auto& src_asic_desc = asic_descriptors_.at(src_asic_id);
            for (const auto& edge : asic.second) {
                auto dst_asic_id = edge.first;
                const auto& dst_asic_desc = asic_descriptors_.at(dst_asic_id);
                for (const auto& eth_conn : edge.second) {
                    EthEndpoint src_id{src_asic_id, eth_conn.src_chan};
                    EthEndpoint dst_id{dst_asic_id, eth_conn.dst_chan};
                    auto connection_key = std::make_pair(std::min(src_id, dst_id), std::max(src_id, dst_id));

                    if (processed_connections.find(connection_key) != processed_connections.end()) {
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

    if (path_to_yaml.has_value()) {
        std::ofstream fout(path_to_yaml.value());
        fout << root;
    } else {
        std::cout << root << std::endl;
    }
}

void PhysicalSystemDescriptor::emit_to_text_proto(const std::optional<std::string>& file_path) {
    emit_physical_system_descriptor_to_text_proto(*this, file_path);
}

void PhysicalSystemDescriptor::validate_graphs() {
    // Validate that the representation of the system is internally consistent.
    for (const auto& [host, asic_group] : system_graph_.asic_connectivity_graph) {
        for (const auto& [src_asic, edges] : asic_group) {
            const auto& src_host = asic_descriptors_.at(src_asic).host_name;
            const auto& src_host_edges = system_graph_.host_connectivity_graph.at(src_host);

            for (const auto& [dst_asic, eth_conns] : edges) {
                const auto& dst_host = asic_descriptors_.at(dst_asic).host_name;

                bool all_local = std::all_of(
                    eth_conns.begin(), eth_conns.end(), [](const EthConnection& conn) { return conn.is_local; });

                bool all_global = std::all_of(
                    eth_conns.begin(), eth_conns.end(), [](const EthConnection& conn) { return !conn.is_local; });

                // All connections must be uniformly local or global.
                TT_FATAL(
                    all_local || all_global,
                    "Physical Discovery Error: All ethernet connections should either be local or global. "
                    "Please reset the system and try again.");

                if (all_local) {
                    // Local connections must remain within the same host.
                    TT_FATAL(
                        src_host == dst_host,
                        "Physical Discovery Error: Local Connection between {} and {} is not on the same host. "
                        "Please reset the system and try again.",
                        src_host,
                        dst_host);
                    continue;  // no need to check further
                }

                // Global connections must cross hosts.
                TT_FATAL(
                    src_host != dst_host,
                    "Physical Discovery Error: Hostnames for connections marked as global should be different. "
                    "Please reset the system and try again.");

                // Validate each global ethernet connection.
                for (const auto& eth_conn : eth_conns) {
                    // Look for a host edge matching dst_host.
                    auto host_edge_it =
                        std::find_if(src_host_edges.begin(), src_host_edges.end(), [&](const auto& host_edge) {
                            return host_edge.first == dst_host;
                        });

                    TT_FATAL(
                        host_edge_it != src_host_edges.end(),
                        "Physical Discovery Error: Global Connection between {} and {} is not found in the host "
                        "connectivity graph. Please reset the system and try again.",
                        src_host,
                        dst_host);

                    const auto& exit_node_conns = host_edge_it->second;
                    bool exit_conn_found = std::any_of(
                        exit_node_conns.begin(), exit_node_conns.end(), [&](const ExitNodeConnection& exit_node_conn) {
                            return exit_node_conn.src_exit_node == src_asic &&
                                   exit_node_conn.dst_exit_node == dst_asic &&
                                   exit_node_conn.eth_conn.src_chan == eth_conn.src_chan &&
                                   exit_node_conn.eth_conn.dst_chan == eth_conn.dst_chan;
                        });

                    TT_FATAL(
                        exit_conn_found,
                        "Physical Discovery Error: Global Connection between {} and {} is not found in the "
                        "host connectivity graph. Please reset the system and try again.",
                        src_host,
                        dst_host);
                }
            }
        }
    }
}

std::vector<AsicID> PhysicalSystemDescriptor::get_asic_neighbors(AsicID asic_id) const {
    for (const auto& [host, asic_group] : system_graph_.asic_connectivity_graph) {
        if (asic_group.find(asic_id) != asic_group.end()) {
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
        if (asic_group.find(src_asic) != asic_group.end()) {
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
    TT_FATAL(
        system_graph_.asic_connectivity_graph.find(hostname) != system_graph_.asic_connectivity_graph.end(),
        "No ASIC topology found for host {}",
        hostname);
    return system_graph_.asic_connectivity_graph.at(hostname);
}

TrayID PhysicalSystemDescriptor::get_tray_id(AsicID asic_id) const {
    TT_FATAL(
        asic_descriptors_.find(asic_id) != asic_descriptors_.end(), "No ASIC descriptor found for asic_id {}", asic_id);
    return asic_descriptors_.at(asic_id).tray_id;
}

ASICLocation PhysicalSystemDescriptor::get_asic_location(AsicID asic_id) const {
    TT_FATAL(
        asic_descriptors_.find(asic_id) != asic_descriptors_.end(), "No ASIC descriptor found for asic_id {}", asic_id);
    return asic_descriptors_.at(asic_id).asic_location;
}

std::vector<AsicID> PhysicalSystemDescriptor::get_asics_connected_to_host(const std::string& hostname) const {
    std::vector<AsicID> asics;
    if (system_graph_.asic_connectivity_graph.find(hostname) != system_graph_.asic_connectivity_graph.end()) {
        for (const auto& [asic_id, _] : system_graph_.asic_connectivity_graph.at(hostname)) {
            asics.push_back(asic_id);
        }
    }
    return asics;
}

std::vector<std::string> PhysicalSystemDescriptor::get_host_neighbors(const std::string& hostname) const {
    TT_FATAL(
        system_graph_.host_connectivity_graph.find(hostname) != system_graph_.host_connectivity_graph.end(),
        "No Host connectivity found for host {}",
        hostname);
    std::vector<std::string> neighbors;
    for (const auto& edge : system_graph_.host_connectivity_graph.at(hostname)) {
        neighbors.push_back(edge.first);
    }
    return neighbors;
}

std::vector<ExitNodeConnection> PhysicalSystemDescriptor::get_connecting_exit_nodes(
    const std::string& src_host, const std::string& dst_host) const {
    TT_FATAL(
        system_graph_.host_connectivity_graph.find(src_host) != system_graph_.host_connectivity_graph.end(),
        "No Host connectivity found for host {}",
        src_host);
    for (const auto& edge : system_graph_.host_connectivity_graph.at(src_host)) {
        if (edge.first == dst_host) {
            return edge.second;
        }
    }
    return {};
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
    if (all_hostnames_unique_) {
        return get_host_name();
    }
    auto my_rank = *(tt::tt_metal::MetalContext::instance().global_distributed_context().rank());
    return get_host_name() + "_" + std::to_string(my_rank);
}

uint32_t PhysicalSystemDescriptor::get_rank_for_hostname(const std::string& host_name) const {
    TT_FATAL(host_to_rank_.find(host_name) != host_to_rank_.end(), "Rank for host {} not found", host_name);
    return host_to_rank_.at(host_name);
}

std::string PhysicalSystemDescriptor::get_host_name_for_asic(AsicID asic_id) const {
    TT_FATAL(
        asic_descriptors_.find(asic_id) != asic_descriptors_.end(), "No ASIC descriptor found for asic_id {}", asic_id);
    return asic_descriptors_.at(asic_id).host_name;
}

UID PhysicalSystemDescriptor::get_u_id(const std::string& hostname) {
    TT_THROW("Querying Host UID requires the Cable Spec which is not currently supported.");
}

RackID PhysicalSystemDescriptor::get_rack_id(const std::string& hostname) {
    TT_THROW("Querying Host Rack ID requires the Cable Spec which is not currently supported.");
}

AisleID PhysicalSystemDescriptor::get_aisle_id(const std::string& hostname) {
    TT_THROW("Querying Host Aisle ID requires the Cable Spec which is not currently supported.");
}

HallID PhysicalSystemDescriptor::get_hall_id(const std::string& hostname) {
    TT_THROW("Querying Host Hall ID requires the Cable Spec which is not currently supported.");
}

}  // namespace tt::tt_metal
