

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

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

}  // namespace

PhysicalSystemDescriptor::PhysicalSystemDescriptor(bool perform_global_discovery, bool run_discovery) {
    if (run_discovery) {
        std::cout << "Running discovery..." << std::endl;
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
        asic_descriptors_[src_unique_id] = ASICDescriptor{tray_id, n_id, cluster_desc->get_board_type(src)};

        std::unordered_map<chip_id_t, size_t> visited_dst;
        // Populate ASIC Graph for Current Host
        for (auto& [chan, dst] : conn) {
            auto dst_chip = std::get<0>(dst);
            auto dst_chan = std::get<1>(dst);
            if (visited_dst.find(dst_chip) == visited_dst.end()) {
                // This neighbor has not been visited. Add it to the graph and mark visited.
                asic_graph[src_unique_id].push_back({chip_unique_ids.at(dst_chip), {EthConnection(chan, dst_chan)}});
                visited_dst[dst_chip] = asic_graph[src_unique_id].size() - 1;
            } else {
                // This neighbor has already been visited. There is more than one channel to it.
                // Update the existing entry with the new channel.
                asic_graph[src_unique_id][visited_dst[dst_chip]].second.push_back(EthConnection(chan, dst_chan));
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
                asic_graph[local_unique_id].push_back({dst_unique_id, {EthConnection(eth_chan, dst_chan)}});
                visited_dst[dst_unique_id] = asic_graph[local_unique_id].size() - 1;
            } else {
                asic_graph[local_unique_id][visited_dst[dst_unique_id]].second.push_back(
                    EthConnection(eth_chan, dst_chan));
            }
            exit_nodes.push_back(ExitNodeConnection{
                .src_exit_node = local_unique_id,
                .dst_exit_node = dst_unique_id,
                .eth_conn = EthConnection(eth_chan, dst_chan)});
        }
    }
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
        this->generate_cross_host_connections();
    }
    this->exchange_metadata(false);
}

}  // namespace tt::tt_metal
