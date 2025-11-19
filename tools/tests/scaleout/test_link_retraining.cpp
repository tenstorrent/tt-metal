// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/control_plane.hpp>
#include "tools/scaleout/validation/utils/cluster_validation_utils.hpp"
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <factory_system_descriptor/utils.hpp>
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include <fmt/format.h>
#include <thread>
#include <chrono>
#include <filesystem>
#include "protobuf/deployment.pb.h"
#include "protobuf/cluster_config.pb.h"
#include "protobuf/node_config.pb.h"
#include "protobuf/factory_system_descriptor.pb.h"

namespace tt::scaleout_tools {

constexpr uint32_t ETH_TRAINING_STATUS_REG = 0x1104;

[[nodiscard]] tt_xy_pair get_eth_core_coord(const tt::Cluster& cluster, ChipId chip_id, uint8_t channel) {
    const auto logical_coord = cluster.get_soc_desc(chip_id).get_eth_core_for_channel(channel, CoordSystem::LOGICAL);
    return cluster.get_virtual_coordinate_from_logical_coordinates(
        chip_id, tt_xy_pair(logical_coord.x, logical_coord.y), CoreType::ETH);
}

void set_link_training_status(const tt::Cluster& cluster, ChipId chip_id, const tt_xy_pair& coord, uint32_t status) {
    const std::vector<uint32_t> status_data{status};
    cluster.write_core(chip_id, coord, status_data, ETH_TRAINING_STATUS_REG);
    cluster.l1_barrier(chip_id);
}

struct TestFixture {
    tt::tt_metal::MetalContext& context;
    const tt::Cluster& cluster;
    const std::unique_ptr<tt::umd::Cluster>& driver;
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> distributed_context;
    tt::tt_metal::PhysicalSystemDescriptor physical_system_descriptor;
    std::unordered_map<uint64_t, ChipId> asic_id_to_chip_id;

    TestFixture() :
        context(tt::tt_metal::MetalContext::instance()),
        cluster(context.get_cluster()),
        driver(cluster.get_driver()),
        distributed_context(context.get_distributed_context_ptr()),
        physical_system_descriptor(driver, distributed_context, &context.hal(), context.rtoptions(), true) {
        for (const auto& [chip_id, asic_id] : cluster.get_unique_chip_ids()) {
            asic_id_to_chip_id[asic_id] = chip_id;
        }
    }
};

[[nodiscard]] uint32_t get_link_training_status(const tt::Cluster& cluster, ChipId chip_id, const tt_xy_pair& coord) {
    std::vector<uint32_t> status(1);
    cluster.read_core(status, sizeof(uint32_t), tt_cxy_pair(chip_id, coord), ETH_TRAINING_STATUS_REG);
    return status[0];
}

void validate_connectivity_after_reset(
    PhysicalSystemDescriptor& physical_system_descriptor, const std::string& gsd_filename_prefix) {
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().global_distributed_context();
    const auto output_path = std::filesystem::temp_directory_path() / "test_link_retraining_validation";
    std::filesystem::create_directories(output_path);

    // Dump the GSD to YAML for validation
    const auto gsd_yaml_path =
        output_path / fmt::format("{}_{}.yaml", gsd_filename_prefix, *distributed_context.rank());
    physical_system_descriptor.dump_to_yaml(gsd_yaml_path.string());
    const auto gsd_yaml_node = YAML::LoadFile(gsd_yaml_path.string());

    // Build FSD from T3K cabling descriptor + hostnames and validate
    constexpr std::string_view cabling_descriptor_path = "tools/tests/scaleout/cabling_descriptors/t3k.textproto";
    const auto hostnames = physical_system_descriptor.get_all_hostnames();

    log_output_rank0("Validating Factory System Descriptor against Global System Descriptor");
    const auto missing_physical_connections = validate_cabling_descriptor_against_gsd(
        std::string(cabling_descriptor_path),
        hostnames,
        gsd_yaml_node,
        true /* strict_validation */,
        true /* fail_on_warning */,
        *distributed_context.rank() == 0 /* log_output */);

    const auto missing_topology =
        generate_asic_topology_from_connections(missing_physical_connections, physical_system_descriptor);

    EXPECT_TRUE(missing_topology.empty()) << "Missing connections detected after link reset";
    log_output_rank0("Factory System Descriptor Validation Complete");

    // Cleanup
    std::filesystem::remove(gsd_yaml_path);
}

// Helper function to process ethernet connections for a given operation
template <typename Operation>
void process_ethernet_connections(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor,
    const std::unordered_map<uint64_t, ChipId>& asic_id_to_chip_id,
    const tt::Cluster& cluster,
    const std::unique_ptr<tt::umd::Cluster>& driver,
    Operation&& operation) {
    const auto cluster_desc = driver->get_cluster_description();
    const auto& asic_topology = physical_system_descriptor.get_asic_topology(physical_system_descriptor.my_host_name());

    for (const auto& [asic_id, asic_connections] : asic_topology) {
        for (const auto& [dst_asic_id, eth_connections] : asic_connections) {
            const auto src_chip_id = asic_id_to_chip_id.at(*asic_id);
            const auto dst_chip_id = asic_id_to_chip_id.at(*dst_asic_id);

            const bool both_mmio =
                cluster_desc->is_chip_mmio_capable(src_chip_id) && cluster_desc->is_chip_mmio_capable(dst_chip_id);
            const bool both_non_mmio =
                !cluster_desc->is_chip_mmio_capable(src_chip_id) && !cluster_desc->is_chip_mmio_capable(dst_chip_id);

            if (both_mmio || both_non_mmio) {
                for (const auto& eth_connection : eth_connections) {
                    operation(src_chip_id, get_eth_core_coord(cluster, src_chip_id, eth_connection.src_chan));
                }
            }
        }
    }
}

TEST(DirectedRetraining, TestActiveEthRetraining) {
    TestFixture fixture;
    validate_connectivity_after_reset(fixture.physical_system_descriptor, "gsd_before_reset");

    // Take down MMIO-to-MMIO and non-MMIO-to-non-MMIO links
    process_ethernet_connections(
        fixture.physical_system_descriptor,
        fixture.asic_id_to_chip_id,
        fixture.cluster,
        fixture.driver,
        [&](ChipId chip_id, const tt_xy_pair& coord) { set_link_training_status(fixture.cluster, chip_id, coord, 0); });

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

    // Validate connectivity after link reset using T3K cabling descriptor
    validate_connectivity_after_reset(fixture.physical_system_descriptor, "gsd_after_reset");
}

TEST(DirectedRetraining, DISABLED_TestExitNodeRetraining) {
    TestFixture fixture;

    const auto& my_hostname = fixture.physical_system_descriptor.my_host_name();
    const auto all_hostnames = fixture.physical_system_descriptor.get_all_hostnames();

    // Take down exit node links to all remote hosts
    for (const auto& host : all_hostnames) {
        if (host == my_hostname) {
            continue;
        }
        const auto exit_nodes = fixture.physical_system_descriptor.get_connecting_exit_nodes(my_hostname, host);
        log_info(tt::LogTest, "Taking {} exit node links down on host {}", exit_nodes.size(), host);

        for (const auto& exit_node : exit_nodes) {
            const auto chip_id = fixture.asic_id_to_chip_id.at(*exit_node.src_exit_node);
            const auto coord = get_eth_core_coord(fixture.cluster, chip_id, exit_node.eth_conn.src_chan);
            set_link_training_status(fixture.cluster, chip_id, coord, 0);
        }
    }

    // Reset all ethernet links
    const auto& asic_topology = fixture.physical_system_descriptor.get_asic_topology(my_hostname);
    reset_ethernet_links(fixture.physical_system_descriptor, asic_topology);

    // Verify exit node links are back up
    for (const auto& host : all_hostnames) {
        if (host == my_hostname) {
            continue;
        }
        const auto exit_nodes = fixture.physical_system_descriptor.get_connecting_exit_nodes(my_hostname, host);

        for (const auto& exit_node : exit_nodes) {
            const auto chip_id = fixture.asic_id_to_chip_id.at(*exit_node.src_exit_node);
            const auto coord = get_eth_core_coord(fixture.cluster, chip_id, exit_node.eth_conn.src_chan);
            EXPECT_EQ(get_link_training_status(fixture.cluster, chip_id, coord), 1);
        }
    }

    fixture.distributed_context->barrier();
    fixture.physical_system_descriptor.run_discovery(true);
}

struct LinkParams {
    std::string host;
    uint32_t tray_id;
    uint32_t asic_location;
    uint32_t channel;
    ChipId chip_id;
    tt_xy_pair coord;
};

[[nodiscard]] std::vector<LinkParams> collect_mmio_link_params(
    const TestFixture& fixture, const tt::tt_metal::AsicTopology& asic_topology) {
    constexpr size_t MAX_LINKS_TO_RESET = 4;

    const auto cluster_desc = fixture.driver->get_cluster_description();
    const auto& asic_descriptors = fixture.physical_system_descriptor.get_asic_descriptors();

    std::vector<LinkParams> local_links;
    std::vector<LinkParams> remote_links;
    local_links.reserve(MAX_LINKS_TO_RESET);
    remote_links.reserve(MAX_LINKS_TO_RESET);

    for (const auto& [asic_id, asic_connections] : asic_topology) {
        const auto src_chip_id = fixture.asic_id_to_chip_id.at(*asic_id);
        if (!cluster_desc->is_chip_mmio_capable(src_chip_id)) {
            continue;
        }

        for (const auto& [dst_id, eth_connections] : asic_connections) {
            if (eth_connections.empty()) {
                continue;
            }

            const auto dst_chip_id = fixture.asic_id_to_chip_id.at(*dst_id);
            if (!cluster_desc->is_chip_mmio_capable(dst_chip_id)) {
                continue;
            }

            const auto& src_asic_desc = asic_descriptors.at(asic_id);
            const auto [dst_asic_id, dst_channel] = fixture.physical_system_descriptor.get_connected_asic_and_channel(
                asic_id, eth_connections.front().src_chan);
            const bool is_local = (src_asic_desc.host_name == asic_descriptors.at(dst_asic_id).host_name);

            const auto src_coord = get_eth_core_coord(fixture.cluster, src_chip_id, eth_connections.front().src_chan);

            const LinkParams link{
                .host = src_asic_desc.host_name,
                .tray_id = *src_asic_desc.tray_id,
                .asic_location = *src_asic_desc.asic_location,
                .channel = eth_connections.front().src_chan,
                .chip_id = src_chip_id,
                .coord = src_coord};

            (is_local ? local_links : remote_links).push_back(link);
        }
    }

    // Select up to MAX_LINKS_TO_RESET links, prioritizing local
    std::vector<LinkParams> selected;
    const auto num_local = std::min(local_links.size(), MAX_LINKS_TO_RESET);
    const auto num_remote = std::min(remote_links.size(), MAX_LINKS_TO_RESET - num_local);

    selected.insert(selected.end(), local_links.begin(), local_links.begin() + num_local);
    selected.insert(selected.end(), remote_links.begin(), remote_links.begin() + num_remote);

    log_info(tt::LogTest, "Found {} links to test ({} local, {} remote)", selected.size(), num_local, num_remote);
    return selected;
}

TEST(DirectedRetraining, TestOnDemandCableRestart) {
    TestFixture fixture;
    validate_connectivity_after_reset(fixture.physical_system_descriptor, "gsd_before_cable_restart");

    const auto& asic_topology =
        fixture.physical_system_descriptor.get_asic_topology(fixture.physical_system_descriptor.my_host_name());
    ASSERT_FALSE(asic_topology.empty()) << "No links available for testing";

    const auto links = collect_mmio_link_params(fixture, asic_topology);
    ASSERT_FALSE(links.empty()) << "No MMIO-to-MMIO links found";

    for (const auto& link : links) {
        log_info(
            tt::LogTest,
            "Testing link: host={}, tray={}, asic={}, channel={}",
            link.host,
            link.tray_id,
            link.asic_location,
            link.channel);

        // Take down the link
        set_link_training_status(fixture.cluster, link.chip_id, link.coord, 0);
        EXPECT_EQ(get_link_training_status(fixture.cluster, link.chip_id, link.coord), 0);

        // Reset the single link we are currently processing
        perform_link_reset(
            link.host, link.tray_id, link.asic_location, link.channel, fixture.physical_system_descriptor);

        // Verify the link is back up
        EXPECT_EQ(get_link_training_status(fixture.cluster, link.chip_id, link.coord), 1);
    }

    // Validate connectivity after link resets using T3K cabling descriptor
    validate_connectivity_after_reset(fixture.physical_system_descriptor, "gsd_after_cable_restart");
}

}  // namespace tt::scaleout_tools
