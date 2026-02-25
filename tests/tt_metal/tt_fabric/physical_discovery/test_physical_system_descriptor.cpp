// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include <iomanip>
#include <map>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include "distributed_context.hpp"
#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/test_utils/test_common.hpp"
#include <llrt/tt_cluster.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>

namespace tt::tt_fabric::physical_discovery {

TEST(PhysicalDiscovery, TestPhysicalSystemDescriptor) {
    using namespace tt::tt_metal::distributed::multihost;
    auto distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    constexpr bool run_discovery = true;

    auto physical_system_desc = tt::tt_metal::PhysicalSystemDescriptor(
        cluster.get_driver(),
        distributed_context,
        &tt::tt_metal::MetalContext::instance().hal(),
        rtoptions,
        run_discovery);
    // Run discovery again to ensure that state is cleared before re-discovery
    physical_system_desc.run_discovery();
    auto hostnames = physical_system_desc.get_all_hostnames();
    // Validate number of hosts discovered
    EXPECT_EQ(hostnames.size(), *(distributed_context->size()));
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

        auto neighbors = physical_system_desc.get_host_neighbors(host);

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
    std::unordered_map<AsicID, ChipId> asic_id_to_chip_id;

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

    if (*(distributed_context->rank()) == 0) {
        // Dump the Generated Physical System Descriptor
        log_info(tt::LogTest, "Dumping Physical System Descriptor to YAML");
        physical_system_desc.dump_to_yaml();
        log_info(tt::LogTest, "Dumping Physical System Descriptor to Text Proto");
        physical_system_desc.emit_to_text_proto();
    }
}

TEST(PhysicalDiscovery, PrintHostTopology) {
    using namespace tt::tt_metal::distributed::multihost;
    auto distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    constexpr bool run_discovery = true;

    auto physical_system_desc = tt::tt_metal::PhysicalSystemDescriptor(
        cluster.get_driver(),
        distributed_context,
        &tt::tt_metal::MetalContext::instance().hal(),
        rtoptions,
        run_discovery);

    if (*(distributed_context->rank()) == 0) {
        auto all_hostnames = physical_system_desc.get_all_hostnames();

        log_info(tt::LogTest, "=== Host Topology ===");
        for (const auto& hostname : all_hostnames) {
            auto host_neighbors = physical_system_desc.get_host_neighbors(hostname);
            std::string neighbors_str = "{";
            for (size_t i = 0; i < host_neighbors.size(); ++i) {
                if (i > 0) {
                    neighbors_str += ", ";
                }
                neighbors_str += host_neighbors[i];
            }
            neighbors_str += "}";
            log_info(tt::LogTest, "{}: {}", hostname, neighbors_str);
        }
        log_info(tt::LogTest, "=== End Host Topology ===");
    }
}

TEST(PhysicalMappingGeneration, Generate2x4SliceToPCIeDeviceMapping) {
    using namespace tt::tt_metal::distributed::multihost;
    auto distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    if (cluster.get_cluster_type() != tt::tt_metal::ClusterType::BLACKHOLE_GALAXY) {
        GTEST_SKIP() << "Splitting a Galaxy into 2x4 Cross-Tray slices is only supported for Blackhole Galaxy Systems.";
    }
    auto physical_system_desc = tt::tt_metal::PhysicalSystemDescriptor(
        cluster.get_driver(), distributed_context, &tt::tt_metal::MetalContext::instance().hal(), rtoptions, true);
    if (*distributed_context->rank() == 0) {
        // A Slice is defined as a 2x4 Grid that spans 2 Trays. Each tray contributes a 2x2 Grid to the slice.
        // Note that this definition corresponds to the tray layout for BH Galaxy Rev A & B
        const std::unordered_map<uint32_t, std::unordered_map<TrayID, std::vector<ASICLocation>>> devices_per_slice = {
            {0,
             {{TrayID{1}, {ASICLocation{1}, ASICLocation{2}, ASICLocation{5}, ASICLocation{6}}},
              {TrayID{3}, {ASICLocation{1}, ASICLocation{2}, ASICLocation{5}, ASICLocation{6}}}}},
            {1,
             {{TrayID{1}, {ASICLocation{3}, ASICLocation{4}, ASICLocation{7}, ASICLocation{8}}},
              {TrayID{3}, {ASICLocation{3}, ASICLocation{4}, ASICLocation{7}, ASICLocation{8}}}}},
            {2,
             {{TrayID{2}, {ASICLocation{3}, ASICLocation{4}, ASICLocation{7}, ASICLocation{8}}},
              {TrayID{4}, {ASICLocation{3}, ASICLocation{4}, ASICLocation{7}, ASICLocation{8}}}}},
            {3,
             {{TrayID{2}, {ASICLocation{1}, ASICLocation{2}, ASICLocation{5}, ASICLocation{6}}},
              {TrayID{4}, {ASICLocation{1}, ASICLocation{2}, ASICLocation{5}, ASICLocation{6}}}}}};
        const auto& pcie_id_to_asic_location = physical_system_desc.get_pcie_id_to_asic_location();
        const auto& pcie_devices_per_tray = physical_system_desc.get_pcie_devices_per_tray();

        YAML::Node slice_to_pcie_device_mapping;
        YAML::Node device_mapping;

        for (const auto& hostname : physical_system_desc.get_all_hostnames()) {
            device_mapping[hostname] = YAML::Node();
            for (const auto& [slice_id, tray_to_asic_location] : devices_per_slice) {
                for (const auto& [tray_id, asic_locations] : tray_to_asic_location) {
                    const auto& pcie_devices = pcie_devices_per_tray.at(hostname).at(*tray_id);
                    for (const auto& pcie_device : pcie_devices) {
                        const auto& asic_location = pcie_id_to_asic_location.at(hostname).at(pcie_device);
                        if (std::find(asic_locations.begin(), asic_locations.end(), asic_location) !=
                            asic_locations.end()) {
                            device_mapping[hostname][slice_id].push_back(pcie_device);
                        }
                    }
                }
            }
        }
        slice_to_pcie_device_mapping["device_mapping"] = device_mapping;
        slice_to_pcie_device_mapping["arch"] = enchantum::to_string(cluster.get_cluster_desc()->get_arch());
        std::ofstream outfile("slice_to_pcie_device_mapping.yaml");
        outfile << slice_to_pcie_device_mapping;
        outfile.close();
    }
}

TEST(PhysicalMappingGeneration, GenerateTrayToPCIeDeviceMapping) {
    using namespace tt::tt_metal::distributed::multihost;
    auto distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();

    auto physical_system_desc = tt::tt_metal::PhysicalSystemDescriptor(
        cluster.get_driver(), distributed_context, &tt::tt_metal::MetalContext::instance().hal(), rtoptions, true);
    const auto& pcie_devices_per_tray = physical_system_desc.get_pcie_devices_per_tray();
    auto my_host = physical_system_desc.my_host_name();

    // Build PCI device ID -> logical ID mapping. UMD now interprets TT_VISIBLE_DEVICES integers as
    // logical IDs (BDF-sorted indices), not PCI device IDs. The cluster descriptor's chip_id is the
    // logical ID; chips_with_mmio maps chip_id (logical) -> pci_device_id.
    std::unordered_map<uint32_t, uint32_t> pcie_id_to_logical_id;
    for (const auto& [logical_id, pcie_id] : cluster.get_cluster_desc()->get_chips_with_mmio()) {
        pcie_id_to_logical_id[static_cast<uint32_t>(pcie_id)] = static_cast<uint32_t>(logical_id);
    }

    // Generate a YAML File with the tray to device mapping (using logical IDs for TT_VISIBLE_DEVICES)
    YAML::Node tray_to_pcie_device_mapping;
    YAML::Node device_mapping;  // Create a separate node for the device mapping
    for (const auto& [tray_id, pcie_devices] : pcie_devices_per_tray.at(my_host)) {
        std::vector<uint32_t> logical_devices_vec;
        for (uint32_t pcie_id : pcie_devices) {
            auto it = pcie_id_to_logical_id.find(pcie_id);
            if (it != pcie_id_to_logical_id.end()) {
                logical_devices_vec.push_back(it->second);
            }
        }
        std::sort(logical_devices_vec.begin(), logical_devices_vec.end());
        device_mapping[tray_id] = logical_devices_vec;
    }
    tray_to_pcie_device_mapping["device_mapping"] = device_mapping;
    tray_to_pcie_device_mapping["arch"] = enchantum::to_string(cluster.get_cluster_desc()->get_arch());
    std::ofstream outfile("tray_to_pcie_device_mapping.yaml");
    outfile << tray_to_pcie_device_mapping;
    outfile.close();
}

}  // namespace tt::tt_fabric::physical_discovery
