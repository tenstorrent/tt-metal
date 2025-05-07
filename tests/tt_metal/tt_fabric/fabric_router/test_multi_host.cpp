// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <filesystem>
#include <memory>
#include <mpi.h>
#include <vector>

#include "fabric_fixture.hpp"
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_fabric {
namespace multi_host_tests {

TEST(MultiHost, TestDualGalaxyControlPlaneInit) {
    tt::tt_metal::detail::InitializeFabricConfig(tt::tt_metal::FabricConfig::FABRIC_2D);
    const std::filesystem::path quanta_galaxy_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(quanta_galaxy_mesh_graph_desc_path.string());
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
}

TEST(MultiHost, TestBasicMPI) {
    MPI_Init(nullptr, nullptr);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_value = rank;
    int global_sum = 0;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Get_processor_name(processor_name, &name_len);

    std::cout << "Hello from processor " << processor_name << ", rank " << rank << " out of " << size << " processes"
              << std::endl;

    MPI_Allreduce(&local_value, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // The expected sum is 0 + 1 + 2 + ... + (size - 1) = size * (size - 1) / 2
    int expected = size * (size - 1) / 2;

    // EXPECT_EQ(global_sum, expected);
    MPI_Finalize();
}

TEST(MultiHost, TestBasicMPICluster) {
    MPI_Init(nullptr, nullptr);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_value = rank;
    int global_sum = 0;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Get_processor_name(processor_name, &name_len);

    std::cout << "Hello from processor " << processor_name << ", rank " << rank << " out of " << size << " processes"
              << std::endl;

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& eth_connections = cluster.get_ethernet_connections();
    std::uint32_t num_expected_chips = 0;
    std::uint32_t num_connections_per_side = 0;
    auto cluster_type = cluster.get_cluster_type();
    if (cluster_type == tt::ClusterType::T3K) {
        num_expected_chips = 8;
        num_connections_per_side = 2;
    } else if (cluster_type == tt::ClusterType::GALAXY) {
        num_expected_chips = 32;
        num_connections_per_side = 4;
    } else {
        GTEST_SKIP() << "Mesh check not supported for system type " << magic_enum::enum_name(cluster_type);
    }
    EXPECT_EQ(eth_connections.size(), num_expected_chips)
        << " Expected " << num_expected_chips << " in " << magic_enum::enum_name(cluster_type) << " cluster";
    for (const auto& [chip, connections] : eth_connections) {
        std::map<chip_id_t, int> num_connections_to_chip;
        for (const auto& [channel, remote_chip_and_channel] : connections) {
            num_connections_to_chip[std::get<0>(remote_chip_and_channel)]++;
        }
        for (const auto& [other_chip, count] : num_connections_to_chip) {
            EXPECT_EQ(count, num_connections_per_side) << "Chip " << chip << " has " << count << " connections to Chip "
                                                       << other_chip << ", expected " << num_connections_per_side;
        }
    }

    MPI_Finalize();
}

}  // namespace multi_host_tests
}  // namespace tt::tt_fabric
