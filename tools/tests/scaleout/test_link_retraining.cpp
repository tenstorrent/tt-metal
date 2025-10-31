// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/control_plane.hpp>
#include "tools/scaleout/validation/utils/cluster_validation_utils.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>

namespace tt::scaleout_tools {

constexpr uint32_t ETH_TRAINING_STATUS_REG = 0x1104;

tt_xy_pair get_eth_core_coord(const tt::Cluster& cluster, ChipId chip_id, uint8_t channel) {
    auto logical_coord = cluster.get_soc_desc(chip_id).get_eth_core_for_channel(channel, CoordSystem::LOGICAL);
    return cluster.get_virtual_coordinate_from_logical_coordinates(
        chip_id, tt_xy_pair(logical_coord.x, logical_coord.y), CoreType::ETH);
}

void set_link_training_status(const tt::Cluster& cluster, ChipId chip_id, const tt_xy_pair& coord, uint32_t status) {
    std::vector<uint32_t> status_vec(1, status);
    cluster.write_core(chip_id, coord, status_vec, ETH_TRAINING_STATUS_REG);
    cluster.l1_barrier(chip_id);
}

struct TestFixture {
    tt::tt_metal::MetalContext& context;
    const tt::Cluster& cluster;
    const std::unique_ptr<tt::umd::Cluster>& driver;
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> distributed_context;
    tt::tt_metal::PhysicalSystemDescriptor physical_system_descriptor;
    std::unordered_map<uint64_t, ChipId> asic_id_to_chip_id;

    TestFixture()
        : context(tt::tt_metal::MetalContext::instance()),
          cluster(context.get_cluster()),
          driver(cluster.get_driver()),
          distributed_context(context.get_distributed_context_ptr()),
          physical_system_descriptor(
              driver,
              distributed_context,
              &context.hal(),
              context.rtoptions(),
              true) {
        for (const auto& [chip_id, asic_id] : cluster.get_unique_chip_ids()) {
            asic_id_to_chip_id[asic_id] = chip_id;
        }
    }



};

uint32_t get_link_training_status(const tt::Cluster& cluster, ChipId chip_id, const tt_xy_pair& coord) {
    std::vector<uint32_t> status(1);
    cluster.read_core(status, sizeof(uint32_t), tt_cxy_pair(chip_id, coord), ETH_TRAINING_STATUS_REG);
    return status[0];
}

// Helper function to process ethernet connections for a given operation
template <typename Operation>
void process_ethernet_connections(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::unordered_map<uint64_t, ChipId>& asic_id_to_chip_id,
    const tt::Cluster& cluster,
    const std::unique_ptr<tt::umd::Cluster>& driver,
    Operation operation) {
    auto cluster_desc = driver->get_cluster_description();
    for (const auto& [asic_id, asic_connections] :
         physical_system_descriptor.get_asic_topology(physical_system_descriptor.my_host_name())) {
        for (const auto& [dst_asic_id, eth_connections] : asic_connections) {
            auto src_chip_id = asic_id_to_chip_id.at(*asic_id);
            auto dst_chip_id = asic_id_to_chip_id.at(*dst_asic_id);
            if ((cluster_desc->is_chip_mmio_capable(src_chip_id) && cluster_desc->is_chip_mmio_capable(dst_chip_id)) ||
                (!cluster_desc->is_chip_mmio_capable(src_chip_id) &&
                 !cluster_desc->is_chip_mmio_capable(dst_chip_id))) {
                for (const auto& eth_connection : eth_connections) {
                    operation(src_chip_id, get_eth_core_coord(cluster, src_chip_id, eth_connection.src_chan));
                }
            }
        }
    }
}

TEST(DirectedRetraining, TestActiveEthRetraining) {
    TestFixture fixture;

    // Take down MMIO-to-MMIO and non-MMIO-to-non-MMIO links
    process_ethernet_connections(
        fixture.physical_system_descriptor,
        fixture.asic_id_to_chip_id,
        fixture.cluster,
        fixture.driver,
        [&](ChipId chip_id, const tt_xy_pair& coord) {
            set_link_training_status(fixture.cluster, chip_id, coord, 0);
        });

    reset_ethernet_links(
        fixture.physical_system_descriptor,
        fixture.physical_system_descriptor.get_asic_topology(fixture.physical_system_descriptor.my_host_name()));

    process_ethernet_connections(
        fixture.physical_system_descriptor,
        fixture.asic_id_to_chip_id,
        fixture.cluster,
        fixture.driver,
        [&](ChipId chip_id, const tt_xy_pair& coord) {
            EXPECT_EQ(get_link_training_status(fixture.cluster, chip_id, coord), 1);
        });

    fixture.physical_system_descriptor.run_discovery(true);
}

TEST(DirectedRetraining, TestExitNodeRetraining) {
    TestFixture fixture;

    for (const auto& host : fixture.physical_system_descriptor.get_all_hostnames()) {
        if (host == fixture.physical_system_descriptor.my_host_name()) {
            continue;
        }
        auto exit_nodes = fixture.physical_system_descriptor.get_connecting_exit_nodes(
            fixture.physical_system_descriptor.my_host_name(), host);
        log_info(tt::LogTest, "Taking {} exit node links down on host {}", exit_nodes.size(), host);

        for (const auto& exit_node : exit_nodes) {
            auto chip_id = fixture.asic_id_to_chip_id.at(*exit_node.src_exit_node);
            auto coord = get_eth_core_coord(fixture.cluster, chip_id, exit_node.eth_conn.src_chan);
            set_link_training_status(fixture.cluster, chip_id, coord, 0);
        }
    }

    reset_ethernet_links(
        fixture.physical_system_descriptor,
        fixture.physical_system_descriptor.get_asic_topology(fixture.physical_system_descriptor.my_host_name()));

    for (const auto& host : fixture.physical_system_descriptor.get_all_hostnames()) {
        if (host == fixture.physical_system_descriptor.my_host_name()) {
            continue;
        }
        auto exit_nodes = fixture.physical_system_descriptor.get_connecting_exit_nodes(
            fixture.physical_system_descriptor.my_host_name(), host);

        for (const auto& exit_node : exit_nodes) {
            auto chip_id = fixture.asic_id_to_chip_id.at(*exit_node.src_exit_node);
            auto coord = get_eth_core_coord(fixture.cluster, chip_id, exit_node.eth_conn.src_chan);
            EXPECT_EQ(get_link_training_status(fixture.cluster, chip_id, coord), 1);
        }
    }

    fixture.distributed_context->barrier();
    fixture.physical_system_descriptor.run_discovery(true);
}

TEST(DirectedRetraining, TestOnDemandCableRestart) {
    TestFixture fixture;

    const auto& asic_topology = fixture.physical_system_descriptor.get_asic_topology(
        fixture.physical_system_descriptor.my_host_name());
    ASSERT_FALSE(asic_topology.empty()) << "No links available for testing";

    auto cluster_desc = fixture.driver->get_cluster_description();
    constexpr size_t MAX_LINKS_TO_TEST = 1;  // System can only handle 1 link safely
    
    struct LinkInfo {
        tt::tt_metal::AsicID src_asic_id;
        tt::tt_metal::AsicID dst_asic_id;
        uint8_t src_channel;
        uint8_t dst_channel;
        ChipId src_chip_id;
        tt_xy_pair src_coord;
        bool is_local;
    };
    
    std::vector<LinkInfo> links;
    links.reserve(MAX_LINKS_TO_TEST);

    // Collect LOCAL links where both chips are MMIO-capable
    // This ensures we don't break remote IO communication paths
    bool found_enough = false;
    for (const auto& [asic_id, asic_connections] : asic_topology) {
        if (found_enough) break;
        auto src_chip_id = fixture.asic_id_to_chip_id.at(*asic_id);
        
        for (const auto& [dst_id, eth_connections] : asic_connections) {
            if (found_enough) break;
            auto dst_chip_id = fixture.asic_id_to_chip_id.at(*dst_id);
            
            // Only test local MMIO-to-MMIO links
            if (cluster_desc->is_chip_mmio_capable(src_chip_id) && cluster_desc->is_chip_mmio_capable(dst_chip_id)) {
                for (const auto& eth_conn : eth_connections) {
                    auto [conn_dst_asic_id, dst_chan] = fixture.physical_system_descriptor.get_connected_asic_and_channel(
                        asic_id, eth_conn.src_chan);
                    const auto& src_asic_desc = fixture.physical_system_descriptor.get_asic_descriptors().at(asic_id);
                    const auto& dst_asic_desc = fixture.physical_system_descriptor.get_asic_descriptors().at(conn_dst_asic_id);
                    bool is_local = (src_asic_desc.host_name == dst_asic_desc.host_name);
                    
                    // Only select LOCAL links to avoid breaking remote communication
                    if (is_local) {
                        links.push_back({
                            asic_id,
                            conn_dst_asic_id,
                            eth_conn.src_chan,
                            dst_chan,
                            src_chip_id,
                            get_eth_core_coord(fixture.cluster, src_chip_id, eth_conn.src_chan),
                            is_local
                        });
                        
                        if (links.size() >= MAX_LINKS_TO_TEST) {
                            found_enough = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    ASSERT_GE(links.size(), 1u) << "No accessible local MMIO-to-MMIO ethernet connections found for testing";
    log_info(tt::LogTest, "Testing {} link reset{}", links.size(), links.size() > 1 ? "s" : "");

    // Test each link sequentially to avoid breaking communication paths
    for (const auto& link : links) {
        log_info(tt::LogTest, "Testing link reset for ASIC {} channel {}", *link.src_asic_id, link.src_channel);
        
        // Take down the link
        set_link_training_status(fixture.cluster, link.src_chip_id, link.src_coord, 0);
        EXPECT_EQ(get_link_training_status(fixture.cluster, link.src_chip_id, link.src_coord), 0) 
            << "Link should be down before reset";

        // Build reset topology for this link
        tt::tt_metal::AsicTopology reset_topology;
        tt::tt_metal::EthConnection src_to_dst_conn{link.src_channel, link.dst_channel, link.is_local};
        tt::tt_metal::EthConnection dst_to_src_conn{link.dst_channel, link.src_channel, link.is_local};
        
        reset_topology[link.src_asic_id].push_back({link.dst_asic_id, {src_to_dst_conn}});
        reset_topology[link.dst_asic_id].push_back({link.src_asic_id, {dst_to_src_conn}});

        // Reset the link
        reset_ethernet_links(fixture.physical_system_descriptor, reset_topology);

        // Verify link is back up
        EXPECT_EQ(get_link_training_status(fixture.cluster, link.src_chip_id, link.src_coord), 1) 
            << "Link should be up after reset";
    }
}

}  // namespace tt::scaleout_tools
