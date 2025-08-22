

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <yaml-cpp/yaml.h>

#include <tt-metalium/distributed_context.hpp>
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/fabric/serialization/intermesh_link_table.hpp"

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

struct EthEndpoint {
    uint64_t board_id;
    uint8_t chan_id;

    auto operator<=>(const EthEndpoint&) const = default;
};

}  // namespace

PhysicalSystemDescriptor::PhysicalSystemDescriptor(bool perform_global_discovery, bool run_discovery) {
    if (run_discovery) {
        run_local_discovery();
        if (perform_global_discovery) {
            run_global_discovery();
        }
    }
}

void PhysicalSystemDescriptor::run_local_discovery() {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();

    const auto& chip_unique_ids = cluster.get_unique_chip_ids();
    const auto& eth_connections = cluster.get_ethernet_connections();
    auto cross_host_eth_connections = cluster.get_ethernet_connections_to_remote_devices();
    auto cluster_desc = cluster.get_cluster_desc();

    auto my_rank = *(distributed_context.rank());
    auto hostname = get_host_name() + "_" + std::to_string(my_rank);
    host_to_mobo_name_[hostname] = get_mobo_name();

    std::set<uint32_t, std::greater<uint32_t>> sorted_pcie_slots = {};
    auto& asic_graph = system_graph_.asic_connectivity_graph[hostname];
    auto& exit_nodes = exit_node_connection_table_[hostname];

    for (const auto& [chip_id, unique_id] : chip_unique_ids) {
        sorted_pcie_slots.insert(cluster.get_physical_slot(chip_id).value());
    }

    for (const auto& [src, conn] : eth_connections) {
        auto src_unique_id = chip_unique_ids.at(src);
        // Populate ASIC Descriptor with Physical Information
        uint32_t n_id = cluster_desc->is_chip_mmio_capable(src) ? 1 : 2;
        uint32_t tray_id =
            1 +
            std::distance(sorted_pcie_slots.begin(), sorted_pcie_slots.find(cluster.get_physical_slot(src).value()));
        asic_descriptors_[src_unique_id] =
            ASICDescriptor{tray_id, n_id, cluster_desc->get_board_type(src), src_unique_id, hostname};

        std::unordered_map<chip_id_t, size_t> visited_dst;
        // Populate ASIC Graph for Current Host
        for (auto& [chan, dst] : conn) {
            auto dst_chip = std::get<0>(dst);
            auto dst_chan = std::get<1>(dst);
            if (visited_dst.find(dst_chip) == visited_dst.end()) {
                // This neighbor has not been visited. Add it to the graph and mark visited.
                asic_graph[src_unique_id].push_back(
                    {chip_unique_ids.at(dst_chip), {EthConnection(chan, dst_chan, true)}});
                visited_dst[dst_chip] = asic_graph[src_unique_id].size() - 1;
            } else {
                // This neighbor has already been visited. There is more than one channel to it.
                // Update the existing entry with the new channel.
                asic_graph[src_unique_id][visited_dst[dst_chip]].second.push_back(EthConnection(chan, dst_chan, true));
            }
        }
    }

    for (const auto& [local_chip_id, eth_link_info] : cross_host_eth_connections) {
        auto local_unique_id = chip_unique_ids.at(local_chip_id);
        std::unordered_map<asic_id_t, size_t> visited_dst;
        for (const auto& [eth_chan, remote_info] : eth_link_info) {
            auto dst_unique_id = std::get<0>(remote_info);
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
    // exit(0);
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
        auto serialized_desc = tt_fabric::serialize_physical_descriptor_to_bytes(*this);
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
            if (rank != controller_rank) {
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
                auto peer_desc = tt_fabric::deserialize_physical_descriptor_from_bytes(serialized_peer_desc);
                this->merge(std::move(peer_desc));
            }
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

void PhysicalSystemDescriptor::run_global_discovery() {
    using namespace tt::tt_metal::distributed::multihost;
    constexpr uint32_t controller_rank = 0;
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    auto my_rank = *(distributed_context.rank());
    this->exchange_metadata(true);
    if (my_rank == controller_rank) {
        this->remove_unresolved_nodes();
        this->generate_cross_host_connections();
    }
    // this->exchange_metadata(false);
    if (my_rank == controller_rank) {
        this->dump_to_yaml("/tmp/physical_system_descriptor.yaml");
    }
    distributed_context.barrier();
}

void PhysicalSystemDescriptor::dump_to_yaml(const std::string& path_to_yaml) {
    YAML::Node root;
    YAML::Node compute_nodes;
    YAML::Node local_eth_connections(YAML::NodeType::Sequence);
    YAML::Node global_eth_connections(YAML::NodeType::Sequence);

    std::set<std::pair<EthEndpoint, EthEndpoint>> processed_connections;
    for (const auto& [host_name, mobo_name] : host_to_mobo_name_) {
        YAML::Node host_node;
        YAML::Node tray_groups;
        host_node["motherboard"] = mobo_name;

        std::unordered_map<tray_id_t, std::vector<ASICDescriptor>> grouped_asics;

        for (const auto& asic : system_graph_.asic_connectivity_graph[host_name]) {
            asic_id_t asic_id = asic.first;
            tray_id_t tray_id = asic_descriptors_.at(asic_id).tray_id;
            grouped_asics[tray_id].push_back(asic_descriptors_.at(asic_id));
        }

        for (const auto& group : grouped_asics) {
            YAML::Node tray_group;
            tray_group["tray_id"] = group.first;  // tray_id
            tray_group["board_type"] = enchantum::to_string(group.second.front().board_type);
            std::vector<ASICDescriptor> sorted_asics = group.second;
            std::sort(sorted_asics.begin(), sorted_asics.end(), [](const ASICDescriptor& a, const ASICDescriptor& b) {
                return a.n_id < b.n_id;
            });
            // Create asics array
            YAML::Node asics_array;
            for (const auto& asic : sorted_asics) {
                YAML::Node asic_node;
                asic_node["nid"] = asic.n_id;
                asic_node["asic_id"] = asic.unique_id;
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
                    src_conn_node["tray_id"] = src_asic_desc.tray_id;
                    src_conn_node["nid"] = src_asic_desc.n_id;
                    dst_conn_node["tray_id"] = dst_asic_desc.tray_id;
                    dst_conn_node["nid"] = dst_asic_desc.n_id;
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

    // for (const auto& )
    std::cout << root << std::endl;
}

}  // namespace tt::tt_metal
