

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed_context.hpp>
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/impl/context/metal_context.hpp"

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

PhysicalSystemDescriptor::PhysicalSystemDescriptor(bool perform_global_discovery) {
    run_local_discovery();
    if (perform_global_discovery) {
        run_global_discovery();
    }
}

void PhysicalSystemDescriptor::run_local_discovery() {
    std::cout << "Running local discovery..." << std::endl;
    using namespace tt::tt_metal::distributed::multihost;

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
        }
    }
}

}  // namespace tt::tt_metal
