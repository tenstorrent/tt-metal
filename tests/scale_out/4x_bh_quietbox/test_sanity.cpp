// SPDX-FileCopyrightText: © 2025 Tenstorrent ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <tt-logger/tt-logger.hpp>
#include <cabling_generator/cabling_generator.hpp>
#include <factory_system_descriptor/utils.hpp>

#include <tt-metalium/control_plane.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric::tests::scale_out {

TEST(Sanity, CorrectPCIeToBusIdMapping) {
    constexpr static size_t NUM_PCI_DEVICES = 4;
    // constexpr static std::array<uint16_t, 4> EXPECTED_PCI_ID_TO_BUS_ID = {0xc1, 0x01, 0x41, 0x42};
    constexpr static std::array<uint16_t, 4> PCI_ID_TO_CHIP_ID = {0, 2, 1, 3};

    const auto& cluster = tt_metal::MetalContext::instance().get_cluster();
    const auto& cluster_desc = cluster.get_cluster_desc();

    auto chip_ids = cluster.all_chip_ids();
    EXPECT_EQ(chip_ids.size(), 4);

    for (size_t pci_id = 0; pci_id < NUM_PCI_DEVICES; ++pci_id) {
        auto chip_id = PCI_ID_TO_CHIP_ID.at(pci_id);
        EXPECT_TRUE(chip_ids.contains(chip_id));

        auto bus_id = cluster.get_bus_id(chip_id);
        // EXPECT_EQ(bus_id, EXPECTED_PCI_ID_TO_BUS_ID.at(pci_id));
        log_info(
            LogTest,
            "PCI ID: 0x{:x}, Bus ID: 0x{:x}, Board ID: 0x{}, UMD Chip ID: {}, UBB ASIC ID: {}",
            pci_id,
            bus_id,
            cluster_desc->get_board_id_for_chip(chip_id),
            chip_id,
            cluster.get_ubb_asic_id(chip_id));
    }
}

TEST(Sanity, ValidateInternalEthernetLinksTrained) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& eth_connections = cluster.get_ethernet_connections();
    log_info(LogTest, "Number of chips with Active Ethernet links: {}", eth_connections.size());

    for (const auto& [chip_id, connected_eth_channels] : eth_connections) {
        for (const auto& [connected_eth_channel, remote_chip_and_channel] : connected_eth_channels) {
            auto [remote_chip, remote_channel] = remote_chip_and_channel;
            log_info(
                LogTest,
                "Chip ID: {} has Active Ethernet ({}) link to Chip ID: {} on Remote Channel ({})",
                chip_id,
                connected_eth_channel,
                remote_chip,
                remote_channel);
        }
    }
}

TEST(Sanity, ReportIntermeshLinks) {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto all_intermesh_links = cluster.get_ethernet_connections_to_remote_devices();

    // Check if cluster supports intermesh links
    if (all_intermesh_links.empty()) {
        log_info(tt::LogTest, "Cluster does not support intermesh links");
        return;
    }

    log_info(tt::LogTest, "Scale-Out Intermesh Link Configuration Report");
    log_info(tt::LogTest, "=============================================");

    // Summary
    size_t total_chips = 0;
    size_t total_links = 0;
    for (const auto& [chip_id, links] : all_intermesh_links) {
        if (!links.empty()) {
            total_chips++;
            total_links += links.size();
        }
    }

    log_info(tt::LogTest, "Total chips with intermesh links: {}", total_chips);
    log_info(tt::LogTest, "Total intermesh links: {}", total_links);
    log_info(tt::LogTest, "");

    // Detailed information per chip
    for (const auto& chip_id : cluster.user_exposed_chip_ids()) {
        if (all_intermesh_links.find(chip_id) != all_intermesh_links.end()) {
            auto links = all_intermesh_links.at(chip_id);
            log_info(tt::LogTest, "Chip {}: {} inter-mesh ethernet links", chip_id, links.size());
            for (const auto& [channel, remote_connection] : links) {
                tt::umd::CoreCoord eth_core =
                    cluster.get_soc_desc(chip_id).get_eth_core_for_channel(channel, CoordSystem::LOGICAL);
                log_info(tt::LogTest, "  Channel {} at {}", channel, CoreCoord{eth_core.x, eth_core.y}.str());
            }
        }
    }
}

TEST(Cluster, TestFactorySystemDescriptor4xBHQuietbox) {
    // Create the cabling generator with file paths
    tt::scaleout_tools::CablingGenerator cabling_generator(
        "tests/scale_out/4x_bh_quietbox/cabling_descriptors/4x_bh_quietbox.textproto",
        "tests/scale_out/4x_bh_quietbox/deployment_descriptors/4x_bh_qb_p150_deployment.textproto");

    // Generate the FSD (textproto format)
    cabling_generator.emit_factory_system_descriptor("fsd/factory_system_descriptor_4x_bh_quietbox.textproto");
    cabling_generator.emit_cabling_guide_csv("fsd/cabling_guide_4x_bh_quietbox.csv");

    // Validate the FSD against the discovered GSD using the common utility function
    tt::scaleout_tools::validate_fsd_against_gsd(
        "fsd/factory_system_descriptor_4x_bh_quietbox.textproto",
        "tests/scale_out/4x_bh_quietbox/global_system_descriptors/4x_bh_quietbox_physical_desc.yaml");
}
}  // namespace tt::tt_fabric::tests::scale_out
