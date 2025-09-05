// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <iomanip>
#include <map>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/control_plane.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/mesh_graph.hpp>
#include "distributed_context.hpp"
#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/test_utils/test_common.hpp"

namespace tt::tt_fabric {
namespace physical_discovery {

TEST(PhysicalDiscovery, TestPhysicalSystemDescriptor) {
    using namespace tt::tt_metal::distributed::multihost;
    auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    auto physical_system_desc = tt::tt_metal::PhysicalSystemDescriptor();
    // Run discovery again to ensure that state is cleared before re-discovery
    physical_system_desc.run_discovery();
    auto hostnames = physical_system_desc.get_all_hostnames();
    // Validate number of hosts discovered
    EXPECT_EQ(hostnames.size(), *(distributed_context.size()));
    // Validate Graph Nodes
    const auto& asic_descs = physical_system_desc.get_asic_descriptors();
    for (const auto& host : hostnames) {
        auto asics = physical_system_desc.get_asics_connected_to_host(host);
        // Ensure that the number of asics discovered per host is consistent
        // with tt_cluster
        EXPECT_EQ(asics.size(), cluster.get_unique_chip_ids().size());

        for (const auto& asic : asics) {
            // Ensure that descriptors were correctly populated for each asic
            EXPECT_NE(asic_descs.find(asic), asic_descs.end());
            EXPECT_EQ(physical_system_desc.get_host_name_for_asic(asic), host);
            for (auto neighbor : physical_system_desc.get_asic_neighbors(asic)) {
                // Ensure that neighbors were correctly populated for each asic
                EXPECT_NE(asic_descs.find(neighbor), asic_descs.end());
            }
        }
        // All to All connectivity for hosts
        auto neighbors = physical_system_desc.get_host_neighbors(host);
        EXPECT_EQ(neighbors.size(), hostnames.size() - 1);

        for (const auto& neighbor : neighbors) {
            EXPECT_NE(std::find(hostnames.begin(), hostnames.end(), neighbor), hostnames.end());
        }
    }

    // Validate Graph Edges
    auto local_eth_links = cluster.get_ethernet_connections();
    auto cross_host_eth_links = cluster.get_ethernet_connections_to_remote_devices();
    auto my_host = physical_system_desc.my_host_name();
    auto my_host_neighbors = physical_system_desc.get_host_neighbors(my_host);

    auto unique_chip_ids = cluster.get_unique_chip_ids();
    std::unordered_map<AsicID, chip_id_t> asic_id_to_chip_id;

    for (const auto& [chip_id, asic_id] : unique_chip_ids) {
        asic_id_to_chip_id[AsicID{asic_id}] = chip_id;
    }

    // Local Connectivity
    for (auto asic : physical_system_desc.get_asics_connected_to_host(my_host)) {
        auto chip_id = asic_id_to_chip_id.at(asic);
        auto eth_links = local_eth_links.at(chip_id);
        auto neighbors = physical_system_desc.get_asic_neighbors(asic);

        for (auto neighbor : neighbors) {
            if (physical_system_desc.get_host_name_for_asic(neighbor) != my_host) {
                // Skip exit nodes
                continue;
            }
            // Ensure that local eth links are populated correctly on the current host
            // This is done by cross referencing eth connectivity returned by the physical
            // descriptor with tt_cluster
            auto dst_chip = asic_id_to_chip_id.at(neighbor);
            auto eth_conns = physical_system_desc.get_eth_connections(asic, neighbor);
            for (const auto& eth_conn : eth_conns) {
                auto [remote_chip, remote_chan] = eth_links.at(eth_conn.src_chan);
                EXPECT_NE(eth_links.find(eth_conn.src_chan), eth_links.end());
                EXPECT_EQ(dst_chip, remote_chip);
                EXPECT_EQ(eth_conn.dst_chan, remote_chan);
            }
        }
    }

    // Host to Host Connectivity
    for (const auto& host : hostnames) {
        if (host == my_host) {
            continue;
        }
        // Ensure that exit nodes are populated correctly on the current host
        // This is done by cross-referencing exit nodes in the physical descriptor with
        // tt_cluster
        auto exit_nodes = physical_system_desc.get_connecting_exit_nodes(my_host, host);
        for (const auto& exit_node : exit_nodes) {
            auto src_asic = exit_node.src_exit_node;
            auto src_chip = asic_id_to_chip_id.at(src_asic);
            auto src_chan = exit_node.eth_conn.src_chan;
            auto dst_asic = exit_node.dst_exit_node;
            auto dst_chan = exit_node.eth_conn.dst_chan;
            auto [remote_asic, remote_chan] = cross_host_eth_links.at(src_chip).at(src_chan);
            auto remote_host = physical_system_desc.get_host_name_for_asic(AsicID{remote_asic});
            // Verify that the exit node asic is marked as a chip with cross host links
            EXPECT_NE(cross_host_eth_links.find(src_chip), cross_host_eth_links.end());
            // Verify that the exit node channel is marked as a cross host link
            EXPECT_NE(cross_host_eth_links.at(src_chip).find(src_chan), cross_host_eth_links.at(src_chip).end());
            // Verify that the remote asic/chan from tt_cluster and the physical descriptor match
            EXPECT_EQ(AsicID{remote_asic}, dst_asic);
            EXPECT_EQ(remote_chan, dst_chan);
            // Verify that remote asic belongs to a neighbor host
            EXPECT_NE(
                std::find(my_host_neighbors.begin(), my_host_neighbors.end(), remote_host), my_host_neighbors.end());
        }
    }

    if (*(distributed_context.rank()) == 0) {
        // Dump the Generated Physical System Descriptor
        log_info(tt::LogTest, "Dumping Physical System Descriptor to YAML");
        physical_system_desc.dump_to_yaml();
        log_info(tt::LogTest, "Dumping Physical System Descriptor to Text Proto");
        physical_system_desc.emit_to_text_proto();
    }
}

}  // namespace physical_discovery
}  // namespace tt::tt_fabric
