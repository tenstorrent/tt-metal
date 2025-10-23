// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/control_plane.hpp>
#include "tools/scaleout/validation/utils/cluster_validation_utils.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <gtest/gtest.h>

namespace tt::scaleout_tools {

TEST(DirectedRetraining, TestActiveEthRetraining) {
    // Run physical discovery and validation to ensure that all links are up
    // Purposely take down active ethernet links on all ASICs
    // Run Ethernet Link retrain API on all ASICs
    // Run discovery and validation again to ensure that all links are up

    auto& context = tt::tt_metal::MetalContext::instance();
    const auto& cluster = context.get_cluster();
    const auto& driver = cluster.get_driver();
    auto distributed_context = context.get_distributed_context_ptr();
    const auto& control_plane = context.get_control_plane();
    std::unordered_set<uint8_t> remote_transfer_eth_chans = {0, 1};
    // driver->configure_active_ethernet_cores_for_mmio_device
    auto physical_system_descriptor = tt::tt_metal::PhysicalSystemDescriptor(
        driver,
        distributed_context,
        &tt::tt_metal::MetalContext::instance().hal(),
        tt::tt_metal::MetalContext::instance().rtoptions().get_mock_enabled(),
        true);

    // Validate that all links are up during initial discovery
    tt_metal::AsicTopology missing_asic_topology = validate_connectivity(
        true,
        true,
        std::filesystem::path("."),
        physical_system_descriptor,
        "tools/tests/scaleout/cabling_descriptors/16_n300_lb_cluster.textproto",
        "tools/tests/scaleout/deployment_descriptors/16_lb_deployment.textproto",
        std::nullopt);
    EXPECT_EQ(missing_asic_topology.size(), 0);

    std::unordered_map<uint64_t, ChipId> asic_id_to_chip_id;
    for (const auto& [chip_id, asic_id] : cluster.get_unique_chip_ids()) {
        asic_id_to_chip_id[asic_id] = chip_id;
    }

    std::vector<uint32_t> zero_vec(1, 0);

    for (const auto& [asic_id, asic_connections] :
         physical_system_descriptor.get_asic_topology(physical_system_descriptor.my_host_name())) {
        for (const auto& [dst_asic_id, eth_connections] : asic_connections) {
            auto src_chip_id = asic_id_to_chip_id[*asic_id];
            auto dst_chip_id = asic_id_to_chip_id[*dst_asic_id];
            for (const auto& eth_connection : eth_connections) {
                auto src_chan = eth_connection.src_chan;
                auto dst_chan = eth_connection.dst_chan;
                auto logical_src_coord =
                    cluster.get_soc_desc(src_chip_id).get_eth_core_for_channel(src_chan, CoordSystem::LOGICAL);
                auto logical_dst_coord =
                    cluster.get_soc_desc(dst_chip_id).get_eth_core_for_channel(dst_chan, CoordSystem::LOGICAL);
                auto src_coord = cluster.get_virtual_coordinate_from_logical_coordinates(
                    src_chip_id, tt_xy_pair(logical_src_coord.x, logical_src_coord.y), CoreType::ETH);
                auto dst_coord = cluster.get_virtual_coordinate_from_logical_coordinates(
                    dst_chip_id, tt_xy_pair(logical_dst_coord.x, logical_dst_coord.y), CoreType::ETH);
                auto src_active_ethernet_cores = control_plane.get_active_ethernet_cores(src_chip_id);
                auto dst_active_ethernet_cores = control_plane.get_active_ethernet_cores(dst_chip_id);

                if (control_plane.get_active_ethernet_cores(src_chip_id, true).find(logical_src_coord) !=
                    control_plane.get_active_ethernet_cores(src_chip_id, true).end()) {
                    EXPECT_NE(
                        control_plane.get_active_ethernet_cores(dst_chip_id, true).find(logical_dst_coord),
                        control_plane.get_active_ethernet_cores(dst_chip_id, true).end());
                    cluster.write_core(src_chip_id, src_coord, zero_vec, 0x1104);
                    cluster.l1_barrier(src_chip_id);
                    cluster.write_core(dst_chip_id, dst_coord, zero_vec, 0x1104);
                    cluster.l1_barrier(dst_chip_id);
                }
            }
        }
    }
    reset_ethernet_links(
        physical_system_descriptor,
        physical_system_descriptor.get_asic_topology(physical_system_descriptor.my_host_name()));
    physical_system_descriptor.run_discovery(true);
    missing_asic_topology = validate_connectivity(
        true,
        true,
        std::filesystem::path("."),
        physical_system_descriptor,
        "tools/tests/scaleout/cabling_descriptors/16_n300_lb_cluster.textproto",
        "tools/tests/scaleout/deployment_descriptors/16_lb_deployment.textproto",
        std::nullopt);
    EXPECT_EQ(missing_asic_topology.size(), 0);
}

}  // namespace tt::scaleout_tools
