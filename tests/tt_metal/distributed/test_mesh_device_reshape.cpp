// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdlib.h>
#include <tt_stl/indestructible.hpp>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include "gmock/gmock.h"
#include <tt-metalium/host_api.hpp>
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/maybe_remote.hpp>
#include "tests/tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "umd/device/types/arch.h"

namespace tt::tt_metal::distributed {
namespace {

using ::testing::SizeIs;

std::vector<chip_id_t> get_physical_device_ids(const MeshDevice& mesh) {
    std::vector<chip_id_t> device_ids;
    for (auto* device : mesh.get_devices()) {
        device_ids.push_back(device->id());
    }
    return device_ids;
}

class T3KReshapeTestFixture : public ::testing::Test {
private:
    inline static ARCH arch = tt::ARCH::Invalid;
    inline static size_t num_devices = 0;

public:
    static void SetUpTestSuite() {
        arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices = tt::tt_metal::GetNumAvailableDevices();
    }

    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            GTEST_SKIP() << "Skipping Multi-Device test suite, since it can only be run in Fast Dispatch Mode.";
        }
        if (num_devices != 8 or arch != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "Skipping T3K Multi-Device test suite on non T3K machine.";
        }
    }
};

std::vector<MeshShape> get_mesh_shapes() {
    static tt::stl::Indestructible<std::vector<MeshShape>> kMeshShapes(std::vector<MeshShape>{
        MeshShape{1, 1}, MeshShape{1, 2}, MeshShape{1, 3}, MeshShape{1, 4}, MeshShape{1, 5}, MeshShape{1, 6},
        MeshShape{1, 7}, MeshShape{1, 8}, MeshShape{2, 1}, MeshShape{2, 2}, MeshShape{2, 3}, MeshShape{2, 4},
        MeshShape{3, 1}, MeshShape{3, 2}, MeshShape{4, 1}, MeshShape{4, 2}, MeshShape{8, 1}, MeshShape{7, 1},
        MeshShape{6, 1}, MeshShape{5, 1}, MeshShape{4, 1}, MeshShape{3, 1}, MeshShape{2, 1}, MeshShape{1, 1}});
    return kMeshShapes.get();
}

class MeshConfigurationTest : public T3KReshapeTestFixture, public ::testing::WithParamInterface<MeshShape> {};

TEST_P(MeshConfigurationTest, MeshConfigurations) {
    const auto& shape = GetParam();
    auto mesh = tt::tt_metal::distributed::MeshDevice::create(
        MeshDeviceConfig(shape),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        tt::tt_metal::DispatchCoreType::WORKER);
    EXPECT_EQ(mesh->shape(), shape);
    mesh->close();
}

TEST_P(MeshConfigurationTest, GetPhysicalDeviceIds) {
    const auto& shape = GetParam();

    auto& system_mesh = SystemMesh::instance();
    EXPECT_THAT(system_mesh.get_mapped_physical_device_ids(shape), SizeIs(shape.mesh_size()));
}

// Test all possible mesh configurations on T3000
INSTANTIATE_TEST_SUITE_P(AllMeshShapes, MeshConfigurationTest, ::testing::ValuesIn(get_mesh_shapes()));

class MeshDeviceReshapeRoundtripTest : public T3KReshapeTestFixture,
                                       public ::testing::WithParamInterface<std::tuple<MeshShape, MeshShape>> {};

TEST_P(MeshDeviceReshapeRoundtripTest, ReshapeBetweenConfigurations) {
    const auto& [old_shape, new_shape] = GetParam();

    if (old_shape.mesh_size() != new_shape.mesh_size()) {
        GTEST_SKIP() << "Device counts don't match; we test this in InvalidReshapeDimensions";
    }
    if (old_shape.is_line_topology() or new_shape.is_line_topology()) {
        GTEST_SKIP() << "Either old or new shape is in line configuration; we test this in From1x4To2x2Invalid";
    }

    auto mesh = tt::tt_metal::distributed::MeshDevice::create(
        MeshDeviceConfig(old_shape),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        tt::tt_metal::DispatchCoreType::WORKER);

    EXPECT_EQ(mesh->shape(), old_shape);

    auto original_order = mesh->get_device_ids();

    // Attempt reshape
    mesh->reshape(new_shape);

    // Verify new shape
    EXPECT_EQ(mesh->shape(), new_shape);

    // Verify device ordering is preserved
    if (old_shape == new_shape) {
        EXPECT_EQ(mesh->get_device_ids(), original_order)
            << "Device ordering is preserved " << MeshShape(old_shape) << " -> " << new_shape;
    } else {
        EXPECT_NE(mesh->get_device_ids(), original_order)
            << "Device ordering is not preserved " << MeshShape(old_shape) << " -> " << new_shape;
    }
}

// Generate all possible combinations of shapes from kMeshShapes
INSTANTIATE_TEST_SUITE_P(
    AllMeshShapes,
    MeshDeviceReshapeRoundtripTest,
    ::testing::Combine(::testing::ValuesIn(get_mesh_shapes()), ::testing::ValuesIn(get_mesh_shapes())));

// Base class for non-parameterized tests
using MeshDeviceReshapeTest = T3KReshapeTestFixture;

TEST_F(MeshDeviceReshapeTest, InvalidRequestedShape) {
    auto& system_mesh = tt::tt_metal::distributed::SystemMesh::instance();

    // Shape too big.
    EXPECT_ANY_THROW(system_mesh.get_mapped_physical_device_ids(MeshShape(9)));
    EXPECT_ANY_THROW(system_mesh.get_mapped_physical_device_ids(MeshShape(2, 5)));

    // Invalid offset.
    EXPECT_ANY_THROW(system_mesh.get_mapped_physical_device_ids(MeshShape(1, 8), /*offset=*/MeshCoordinate(0, 1)));
    EXPECT_ANY_THROW(system_mesh.get_mapped_physical_device_ids(MeshShape(2, 3), /*offset=*/MeshCoordinate(1, 1)));

    // Offset dimensionality mismatch.
    EXPECT_ANY_THROW(system_mesh.get_mapped_physical_device_ids(MeshShape(2, 3), /*offset=*/MeshCoordinate(1)));

    // Mismatch system mesh shape.
    EXPECT_ANY_THROW(system_mesh.get_mapped_physical_device_ids(MeshShape(8), /*offset=*/MeshCoordinate(1)));
}

TEST_F(MeshDeviceReshapeTest, InvalidReshapeDimensions) {
    auto mesh = tt::tt_metal::distributed::MeshDevice::create(
        MeshDeviceConfig(MeshShape(1, 8)),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        tt::tt_metal::DispatchCoreType::WORKER);

    // Test reshaping to dimensions that don't match total device count
    EXPECT_THROW(mesh->reshape(MeshShape(3, 3)), std::runtime_error);  // 9 devices != 8
    EXPECT_THROW(mesh->reshape(MeshShape(1, 9)), std::runtime_error);  // 9 devices != 8

    // Verify original shape is preserved after failed reshapes
    EXPECT_EQ(mesh->shape(), MeshShape(1, 8));
}

TEST_F(MeshDeviceReshapeTest, From1x8To2x4ThenBackTo1x8) {
    auto mesh = tt::tt_metal::distributed::MeshDevice::create(
        MeshDeviceConfig(MeshShape(1, 8)),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        tt::tt_metal::DispatchCoreType::WORKER);

    EXPECT_EQ(mesh->shape(), MeshShape(1, 8));
    auto original_order = mesh->get_device_ids();

    mesh->reshape(MeshShape(2, 4));

    EXPECT_EQ(mesh->shape(), MeshShape(2, 4));
    std::vector<chip_id_t> expected_physical_device_id_order = {
        original_order[0],
        original_order[1],
        original_order[2],
        original_order[3],
        original_order[7],
        original_order[6],
        original_order[5],
        original_order[4],
    };

    auto new_order = mesh->get_device_ids();
    EXPECT_EQ(new_order, expected_physical_device_id_order);

    mesh->reshape(MeshShape(1, 8));
    EXPECT_EQ(mesh->get_device_ids(), original_order);
}

TEST_F(MeshDeviceReshapeTest, InvalidTotalDeviceCount) {
    auto mesh = tt::tt_metal::distributed::MeshDevice::create(
        MeshDeviceConfig(MeshShape(1, 8)),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        tt::tt_metal::DispatchCoreType::WORKER);

    // Test reshaping to dimensions that don't match total device count
    EXPECT_THROW(mesh->reshape(MeshShape(3, 3)), std::runtime_error);  // 9 devices != 8
    EXPECT_THROW(mesh->reshape(MeshShape(1, 9)), std::runtime_error);  // 9 devices != 8

    // Verify original shape is preserved after failed reshapes
    EXPECT_EQ(mesh->shape(), MeshShape(1, 8));
}

TEST_F(MeshDeviceReshapeTest, From1x4To2x2Invalid) {
    auto mesh = tt::tt_metal::distributed::MeshDevice::create(
        MeshDeviceConfig(MeshShape(1, 4)),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        tt::tt_metal::DispatchCoreType::WORKER);

    // This is an invalid reshape because the 1x4 mesh does not fully cover the 2x2 mesh
    EXPECT_THROW(mesh->reshape(MeshShape(2, 2)), std::runtime_error);
}

TEST_F(MeshDeviceReshapeTest, From1x4To2x2Valid) {
    auto& system_mesh = tt::tt_metal::distributed::SystemMesh::instance();

    // Fetch the device ids for a physically connected 2x2 mesh.
    auto physical_device_ids = system_mesh.get_mapped_physical_device_ids(MeshShape(2, 2));

    // Supply the physical device ids to the mesh constructor that we know we know is 2x2 physically connected.
    // We will create a 1x4 mesh and then reshape it to 2x2.
    auto mesh = tt::tt_metal::distributed::MeshDevice::create(
        MeshDeviceConfig(MeshShape(1, 4), /*offset=*/std::nullopt, /*physical_device_ids=*/physical_device_ids),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        tt::tt_metal::DispatchCoreType::WORKER);

    mesh->reshape(MeshShape(2, 2));
    EXPECT_EQ(mesh->shape(), MeshShape(2, 2));
    auto new_layout = mesh->get_device_ids();
    for (auto physical_device_id : physical_device_ids) {
        EXPECT_TRUE(std::find(new_layout.begin(), new_layout.end(), physical_device_id) != new_layout.end());
    }
}

TEST_F(MeshDeviceReshapeTest, From2x2To1x4) {
    auto mesh = tt::tt_metal::distributed::MeshDevice::create(
        MeshDeviceConfig(MeshShape(2, 2)),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        tt::tt_metal::DispatchCoreType::WORKER);

    auto mesh_2x2_device_ids = mesh->get_device_ids();

    mesh->reshape(MeshShape(1, 4));
    EXPECT_EQ(mesh->shape(), MeshShape(1, 4));

    auto mesh_1x4_device_ids = mesh->get_device_ids();
    std::vector<chip_id_t> expected_1x4_device_ids = {
        mesh_2x2_device_ids[0],
        mesh_2x2_device_ids[1],
        mesh_2x2_device_ids[3],
        mesh_2x2_device_ids[2],
    };

    EXPECT_EQ(mesh_1x4_device_ids, expected_1x4_device_ids);
}

}  // namespace
}  // namespace tt::tt_metal::distributed
