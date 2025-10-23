// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/control_plane.hpp>
#include "tools/scaleout/validation/utils/cluster_validation_utils.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <gtest/gtest.h>

namespace tt::scaleout_tools {

// Helper function to process ethernet connections for a given operation
template <typename Operation>
void process_ethernet_connections(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::unordered_map<uint64_t, ChipId>& asic_id_to_chip_id,
    const tt::Cluster& cluster,
    const tt::umd::ClusterDescriptor* cluster_desc,
    Operation operation) {
    for (const auto& [asic_id, asic_connections] :
         physical_system_descriptor.get_asic_topology(physical_system_descriptor.my_host_name())) {
        for (const auto& [dst_asic_id, eth_connections] : asic_connections) {
            auto src_chip_id = asic_id_to_chip_id.at(*asic_id);
            auto dst_chip_id = asic_id_to_chip_id.at(*dst_asic_id);
            if ((cluster_desc->is_chip_mmio_capable(src_chip_id) && cluster_desc->is_chip_mmio_capable(dst_chip_id)) ||
                (!cluster_desc->is_chip_mmio_capable(src_chip_id) &&
                 !cluster_desc->is_chip_mmio_capable(dst_chip_id))) {
                for (const auto& eth_connection : eth_connections) {
                    auto src_chan = eth_connection.src_chan;
                    auto logical_src_coord =
                        cluster.get_soc_desc(src_chip_id).get_eth_core_for_channel(src_chan, CoordSystem::LOGICAL);
                    auto src_coord = cluster.get_virtual_coordinate_from_logical_coordinates(
                        src_chip_id, tt_xy_pair(logical_src_coord.x, logical_src_coord.y), CoreType::ETH);

                    operation(src_chip_id, src_coord);
                }
            }
        }
    }
}

TEST(DirectedRetraining, TestActiveEthRetraining) {
    // Run physical discovery.
    // Purposely take down active ethernet links on specific ASICs.
    // Run Ethernet Link retrain API on all ASICs
    // Readback ethernet link status from all ASICs and ensure that the links are retrained successfully.

    auto& context = tt::tt_metal::MetalContext::instance();
    const auto& cluster = context.get_cluster();
    const auto& driver = cluster.get_driver();
    auto distributed_context = context.get_distributed_context_ptr();
    // Discover all links in the cluster.
    auto physical_system_descriptor = tt::tt_metal::PhysicalSystemDescriptor(
        driver,
        distributed_context,
        &tt::tt_metal::MetalContext::instance().hal(),
        tt::tt_metal::MetalContext::instance().rtoptions().get_mock_enabled(),
        true);

    std::unordered_map<uint64_t, ChipId> asic_id_to_chip_id;
    for (const auto& [chip_id, asic_id] : cluster.get_unique_chip_ids()) {
        asic_id_to_chip_id[asic_id] = chip_id;
    }

    // Set training status on specific links to 0 (LINK_TRAIN_FAIL) to purposely take down them down
    // Logic for selecting which links to take down is as follows:
    // Take down eth links connecting non-MMIO chips to non-MMIO chips
    // Take down eth links connecting MMIO chips to MMIO chips
    // On a single T3K this will take down 12 / 20 links in the cluster
    // On a single 6U Galaxy this will take down all links in the cluster
    std::vector<uint32_t> zero_vec(1, 0);
    auto cluster_desc = driver->get_cluster_description();
    process_ethernet_connections(
        physical_system_descriptor,
        asic_id_to_chip_id,
        cluster,
        cluster_desc,
        [&cluster, &zero_vec](ChipId src_chip_id, const tt_xy_pair& src_coord) {
            cluster.write_core(src_chip_id, src_coord, zero_vec, 0x1104);
            cluster.l1_barrier(src_chip_id);
        });

    // Reset all links in the cluster
    reset_ethernet_links(
        physical_system_descriptor,
        physical_system_descriptor.get_asic_topology(physical_system_descriptor.my_host_name()));

    // Verify that links retraining was successful. All links should report a status of 1 (LINK_TRAIN_SUCCESS)
    process_ethernet_connections(
        physical_system_descriptor,
        asic_id_to_chip_id,
        cluster,
        cluster_desc,
        [&cluster](ChipId src_chip_id, const tt_xy_pair& src_coord) {
            std::vector<uint32_t> src_value = {0};
            cluster.read_core(src_value, sizeof(uint32_t), tt_cxy_pair(src_chip_id, src_coord), 0x1104);
            EXPECT_EQ(src_value[0], 1);
        });

    // Run discovery and validate connectivity again
    physical_system_descriptor.run_discovery(true);
}

TEST(DirectedRetraining, ExitNodeRetraining) {
    // Run physical discovery on all compute nodes in the cluster.
    // Purposely take down all exit node links.
    // Issue retrain API on all liks.
    // Readback ethernet link status from all exit nodes and ensure that the links are retrained successfully.

    auto& context = tt::tt_metal::MetalContext::instance();
    const auto& cluster = context.get_cluster();
    const auto& driver = cluster.get_driver();
    auto distributed_context = context.get_distributed_context_ptr();

    std::unordered_map<uint64_t, ChipId> asic_id_to_chip_id;
    for (const auto& [chip_id, asic_id] : cluster.get_unique_chip_ids()) {
        asic_id_to_chip_id[asic_id] = chip_id;
    }

    // Discover all links in the cluster.
    auto physical_system_descriptor = tt::tt_metal::PhysicalSystemDescriptor(
        driver,
        distributed_context,
        &tt::tt_metal::MetalContext::instance().hal(),
        tt::tt_metal::MetalContext::instance().rtoptions().get_mock_enabled(),
        true);

    // Loop over all neighboring hosts
    // For each neighboring host, query the exit nodes connecting to the current host
    // For each exit node connection, take down the link by writing 0 to the link training status register
    std::vector<uint32_t> zero_vec(1, 0);
    for (const auto& host : physical_system_descriptor.get_all_hostnames()) {
        if (host == physical_system_descriptor.my_host_name()) {
            continue;
        }
        auto exit_nodes =
            physical_system_descriptor.get_connecting_exit_nodes(physical_system_descriptor.my_host_name(), host);
        log_info(tt::LogTest, "Take {} exit node links down on host {}", exit_nodes.size(), host);
        for (const auto& exit_node : exit_nodes) {
            auto src_chan = exit_node.eth_conn.src_chan;
            auto logical_src_coord = cluster.get_soc_desc(asic_id_to_chip_id.at(*exit_node.src_exit_node))
                                         .get_eth_core_for_channel(src_chan, CoordSystem::LOGICAL);
            auto src_coord = cluster.get_virtual_coordinate_from_logical_coordinates(
                asic_id_to_chip_id.at(*exit_node.src_exit_node),
                tt_xy_pair(logical_src_coord.x, logical_src_coord.y),
                CoreType::ETH);
            cluster.write_core(asic_id_to_chip_id.at(*exit_node.src_exit_node), src_coord, zero_vec, 0x1104);
        }
    }

    // Issue retrain API on all links in the cluster
    reset_ethernet_links(
        physical_system_descriptor,
        physical_system_descriptor.get_asic_topology(physical_system_descriptor.my_host_name()));

    // Verify that exit node link retraining was successful. All links should report a status of 1 (LINK_TRAIN_SUCCESS)
    for (const auto& host : physical_system_descriptor.get_all_hostnames()) {
        if (host == physical_system_descriptor.my_host_name()) {
            continue;
        }
        auto exit_nodes =
            physical_system_descriptor.get_connecting_exit_nodes(physical_system_descriptor.my_host_name(), host);
        for (const auto& exit_node : exit_nodes) {
            auto src_chan = exit_node.eth_conn.src_chan;
            auto logical_src_coord = cluster.get_soc_desc(asic_id_to_chip_id.at(*exit_node.src_exit_node))
                                         .get_eth_core_for_channel(src_chan, CoordSystem::LOGICAL);
            auto src_coord = cluster.get_virtual_coordinate_from_logical_coordinates(
                asic_id_to_chip_id.at(*exit_node.src_exit_node),
                tt_xy_pair(logical_src_coord.x, logical_src_coord.y),
                CoreType::ETH);
            std::vector<uint32_t> src_value = {0};
            cluster.read_core(
                src_value,
                sizeof(uint32_t),
                tt_cxy_pair(asic_id_to_chip_id.at(*exit_node.src_exit_node), src_coord),
                0x1104);
            EXPECT_EQ(src_value[0], 1);
        }
    }
    distributed_context->barrier();
    physical_system_descriptor.run_discovery(true);
}

}  // namespace tt::scaleout_tools
