// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <umd/device/coordinates/coordinate_manager.hpp>

#include "impl/context/metal_env_accessor.hpp"
#include "impl/device/mock_device_util.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal {

TEST(L2cpuCoordinates, BlackholeMesh) {
    MetalEnv env{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::BLACKHOLE, 2))};
    auto mesh = env.create_mesh_device(distributed::MeshDeviceConfig(env.get_system_mesh().shape()));
    constexpr std::array<uint32_t, 4> expected_y{3, 5, 7, 9};
    for (uint32_t logical_y = 0; logical_y < expected_y.size(); ++logical_y) {
        EXPECT_EQ(
            mesh->virtual_core_from_logical_core(CoreCoord(0, logical_y), CoreType::L2CPU),
            CoreCoord(8, expected_y[logical_y]));
    }
    EXPECT_THROW(mesh->virtual_core_from_logical_core(CoreCoord(1, 0), CoreType::L2CPU), std::runtime_error);
    EXPECT_THROW(mesh->virtual_core_from_logical_core(CoreCoord(0, 4), CoreType::L2CPU), std::runtime_error);
    EXPECT_THROW(mesh->virtual_core_from_logical_core(CoreCoord(0, 0), CoreType::PCIE), std::runtime_error);
    EXPECT_EQ(
        mesh->virtual_core_from_logical_core(CoreCoord(0, 0), CoreType::WORKER),
        mesh->virtual_core_from_logical_core(CoreCoord(0, 0), CoreType::TENSIX));
}

TEST(L2cpuCoordinates, HarvestedLogicalOrder) {
    tt::HarvestingMasks masks{};
    masks.eth_harvesting_mask = 0x120;
    masks.pcie_harvesting_mask = 0x2;
    masks.l2cpu_harvesting_mask = 0b0010;
    auto manager = tt::umd::CoordinateManager::create_coordinate_manager(tt::ARCH::BLACKHOLE, true, masks);
    constexpr std::array<uint32_t, 3> expected_y{3, 7, 9};
    for (uint32_t logical_y = 0; logical_y < expected_y.size(); ++logical_y) {
        const auto coord =
            manager->translate_coord_to({0, logical_y, CoreType::L2CPU, CoordSystem::LOGICAL}, CoordSystem::TRANSLATED);
        EXPECT_EQ(coord.x, 8);
        EXPECT_EQ(coord.y, expected_y[logical_y]);
    }
    EXPECT_THROW(
        manager->translate_coord_to({0, 3, CoreType::L2CPU, CoordSystem::LOGICAL}, CoordSystem::TRANSLATED),
        std::runtime_error);
}

TEST(L2cpuCoordinates, UnavailableOnWormhole) {
    MetalEnv env{MetalEnvDescriptor(experimental::get_mock_cluster_desc_name(tt::ARCH::WORMHOLE_B0, 1))};
    const auto& cluster = MetalEnvAccessor(env).impl().get_cluster();
    ASSERT_FALSE(cluster.all_chip_ids().empty());
    EXPECT_THROW(
        cluster.get_virtual_coordinate_from_logical_coordinates(
            *cluster.all_chip_ids().begin(), CoreCoord(0, 0), CoreType::L2CPU),
        std::runtime_error);
}

}  // namespace tt::tt_metal
