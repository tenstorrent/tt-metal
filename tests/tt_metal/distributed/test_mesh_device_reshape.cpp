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
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <umd/device/types/arch.hpp>

namespace tt::tt_metal::distributed {
namespace {

using ::testing::ElementsAre;
using ::testing::SizeIs;

std::vector<chip_id_t> get_physical_device_ids(const MeshDevice& mesh) {
    std::vector<chip_id_t> device_ids;
    for (auto* device : mesh.get_devices()) {
        device_ids.push_back(device->id());
    }
    return device_ids;
}

std::vector<MeshShape> get_mesh_shapes() {
    static tt::stl::Indestructible<std::vector<MeshShape>> kMeshShapes(std::vector<MeshShape>{
        MeshShape{1, 1}, MeshShape{1, 2}, MeshShape{1, 3}, MeshShape{1, 4}, MeshShape{1, 5}, MeshShape{1, 6},
        MeshShape{1, 7}, MeshShape{1, 8}, MeshShape{2, 1}, MeshShape{2, 2}, MeshShape{2, 3}, MeshShape{2, 4},
        MeshShape{3, 1}, MeshShape{3, 2}, MeshShape{4, 1}, MeshShape{4, 2}, MeshShape{8, 1}, MeshShape{7, 1},
        MeshShape{6, 1}, MeshShape{5, 1}, MeshShape{4, 1}, MeshShape{3, 1}, MeshShape{2, 1}, MeshShape{1, 1}});
    return kMeshShapes.get();
}

class MeshConfigurationTest : public MeshDeviceFixtureBase, public ::testing::WithParamInterface<MeshShape> {
public:
    MeshConfigurationTest() :
        MeshDeviceFixtureBase(Config{
            .mesh_shape = GetParam(),
        }) {}
};

TEST_P(MeshConfigurationTest, MeshConfigurations) { EXPECT_EQ(mesh_device_->shape(), GetParam()); }

TEST_P(MeshConfigurationTest, GetMappedDevices) {
    const auto& shape = GetParam();

    auto& system_mesh = SystemMesh::instance();
    EXPECT_THAT(system_mesh.get_mapped_devices(shape).device_ids, SizeIs(shape.mesh_size()));
    EXPECT_THAT(system_mesh.get_mapped_devices(shape).fabric_node_ids, SizeIs(shape.mesh_size()));
}

// Test all possible mesh configurations on T3000
INSTANTIATE_TEST_SUITE_P(AllMeshShapes, MeshConfigurationTest, ::testing::ValuesIn(get_mesh_shapes()));

class MeshDeviceReshapeRoundtripTest : public MeshDeviceFixtureBase,
                                       public ::testing::WithParamInterface<std::tuple<MeshShape, MeshShape>> {
public:
    MeshDeviceReshapeRoundtripTest() :
        MeshDeviceFixtureBase(Config{
            .mesh_shape = std::get<0>(GetParam()),
        }) {}
};

TEST_P(MeshDeviceReshapeRoundtripTest, ReshapeBetweenConfigurations) {
    const auto& [old_shape, new_shape] = GetParam();

    if (old_shape.mesh_size() != new_shape.mesh_size()) {
        GTEST_SKIP() << "Device counts don't match; we test this in InvalidReshapeDimensions";
    }
    if (old_shape.is_line_topology() or new_shape.is_line_topology()) {
        GTEST_SKIP() << "Either old or new shape is in line configuration; we test this in From1x4To2x2Invalid";
    }

    EXPECT_EQ(mesh_device_->shape(), old_shape);

    auto original_order = mesh_device_->get_device_ids();

    // Attempt reshape
    mesh_device_->reshape(new_shape);

    // Verify new shape
    EXPECT_EQ(mesh_device_->shape(), new_shape);

    // Verify device ordering is preserved
    if (old_shape == new_shape) {
        EXPECT_EQ(mesh_device_->get_device_ids(), original_order)
            << "Device ordering is preserved " << MeshShape(old_shape) << " -> " << new_shape;
    } else {
        EXPECT_NE(mesh_device_->get_device_ids(), original_order)
            << "Device ordering is not preserved " << MeshShape(old_shape) << " -> " << new_shape;
    }
}

// Generate all possible combinations of shapes from kMeshShapes
INSTANTIATE_TEST_SUITE_P(
    AllMeshShapes,
    MeshDeviceReshapeRoundtripTest,
    ::testing::Combine(::testing::ValuesIn(get_mesh_shapes()), ::testing::ValuesIn(get_mesh_shapes())));

class MeshDevice1x8ReshapeTest : public MeshDeviceFixtureBase {
public:
    MeshDevice1x8ReshapeTest() :
        MeshDeviceFixtureBase(Config{
            .mesh_shape = MeshShape{1, 8},
        }) {}
};

TEST_F(MeshDevice1x8ReshapeTest, InvalidRequestedShape) {
    auto& system_mesh = tt::tt_metal::distributed::SystemMesh::instance();

    // Shape too big.
    EXPECT_ANY_THROW(system_mesh.get_mapped_devices(MeshShape(9)));
    EXPECT_ANY_THROW(system_mesh.get_mapped_devices(MeshShape(2, 5)));

    // Invalid offset.
    EXPECT_ANY_THROW(system_mesh.get_mapped_devices(MeshShape(1, 8), /*offset=*/MeshCoordinate(0, 1)));
    EXPECT_ANY_THROW(system_mesh.get_mapped_devices(MeshShape(2, 3), /*offset=*/MeshCoordinate(1, 1)));

    // Offset dimensionality mismatch.
    EXPECT_ANY_THROW(system_mesh.get_mapped_devices(MeshShape(2, 3), /*offset=*/MeshCoordinate(1)));

    // Mismatch system mesh shape.
    EXPECT_ANY_THROW(system_mesh.get_mapped_devices(MeshShape(8), /*offset=*/MeshCoordinate(1)));
}

TEST_F(MeshDevice1x8ReshapeTest, InvalidReshapeDimensions) {
    // Test reshaping to dimensions that don't match total device count
    EXPECT_THROW(mesh_device_->reshape(MeshShape(3, 3)), std::runtime_error);  // 9 devices != 8
    EXPECT_THROW(mesh_device_->reshape(MeshShape(1, 9)), std::runtime_error);  // 9 devices != 8

    // Verify original shape is preserved after failed reshapes
    EXPECT_EQ(mesh_device_->shape(), MeshShape(1, 8));
}

TEST_F(MeshDevice1x8ReshapeTest, From1x8To2x4ThenBackTo1x8) {
    EXPECT_EQ(mesh_device_->shape(), MeshShape(1, 8));
    auto original_order = mesh_device_->get_device_ids();

    mesh_device_->reshape(MeshShape(2, 4));

    EXPECT_EQ(mesh_device_->shape(), MeshShape(2, 4));
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

    auto new_order = mesh_device_->get_device_ids();
    EXPECT_EQ(new_order, expected_physical_device_id_order);

    mesh_device_->reshape(MeshShape(1, 8));
    EXPECT_EQ(mesh_device_->get_device_ids(), original_order);
}

TEST_F(MeshDevice1x8ReshapeTest, InvalidTotalDeviceCount) {
    // Test reshaping to dimensions that don't match total device count
    EXPECT_THROW(mesh_device_->reshape(MeshShape(3, 3)), std::runtime_error);  // 9 devices != 8
    EXPECT_THROW(mesh_device_->reshape(MeshShape(1, 9)), std::runtime_error);  // 9 devices != 8

    // Verify original shape is preserved after failed reshapes
    EXPECT_EQ(mesh_device_->shape(), MeshShape(1, 8));
}

class MeshDevice1x4ReshapeTest : public MeshDeviceFixtureBase {
public:
    MeshDevice1x4ReshapeTest() :
        MeshDeviceFixtureBase(Config{
            .mesh_shape = MeshShape{1, 4},
        }) {}
};

TEST_F(MeshDevice1x4ReshapeTest, From1x4To2x2Invalid) {
    // This is an invalid reshape because the 1x4 mesh does not fully cover the 2x2 mesh
    EXPECT_THROW(mesh_device_->reshape(MeshShape(2, 2)), std::runtime_error);
}

class MeshDevice2x2ReshapeTest : public MeshDeviceFixtureBase {
public:
    MeshDevice2x2ReshapeTest() :
        MeshDeviceFixtureBase(Config{
            .mesh_shape = MeshShape{2, 2},
        }) {}
};

TEST_F(MeshDevice2x2ReshapeTest, From1x4To2x2Valid) {
    // Fetch the device ids for a physically connected 2x2 mesh.
    EXPECT_EQ(mesh_device_->shape(), MeshShape(2, 2));
    std::vector<chip_id_t> physical_device_ids = mesh_device_->get_device_ids();

    // Supply the physical device ids to the mesh constructor that we know we know is 2x2 physically connected.
    // We will create a 1x4 mesh and then reshape it to 2x2.
    mesh_device_.reset();
    mesh_device_ = tt::tt_metal::distributed::MeshDevice::create(
        MeshDeviceConfig(MeshShape(1, 4), /*offset=*/std::nullopt, /*physical_device_ids=*/physical_device_ids),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        tt::tt_metal::DispatchCoreType::WORKER);
    EXPECT_EQ(mesh_device_->shape(), MeshShape(1, 4));

    mesh_device_->reshape(MeshShape(2, 2));
    EXPECT_EQ(mesh_device_->shape(), MeshShape(2, 2));

    EXPECT_THAT(
        mesh_device_->get_device_ids(),
        ElementsAre(physical_device_ids[0], physical_device_ids[1], physical_device_ids[2], physical_device_ids[3]));
}

class MeshDevice2x2WithOffsetReshapeTest : public MeshDeviceFixtureBase,
                                           public testing::WithParamInterface<MeshCoordinate> {
public:
    MeshDevice2x2WithOffsetReshapeTest() :
        MeshDeviceFixtureBase(Config{
            .mesh_shape = MeshShape{2, 2},
            .mesh_offset = GetParam(),
        }) {}
};

TEST_P(MeshDevice2x2WithOffsetReshapeTest, From2x2To1x4ThenBackTo2x2) {
    auto mesh_2x2_device_ids = mesh_device_->get_device_ids();

    mesh_device_->reshape(MeshShape(1, 4));
    EXPECT_EQ(mesh_device_->shape(), MeshShape(1, 4));

    EXPECT_THAT(
        mesh_device_->get_device_ids(),
        ElementsAre(mesh_2x2_device_ids[0], mesh_2x2_device_ids[1], mesh_2x2_device_ids[3], mesh_2x2_device_ids[2]));

    mesh_device_->reshape(MeshShape(2, 2));
    EXPECT_EQ(mesh_device_->shape(), MeshShape(2, 2));

    EXPECT_THAT(
        mesh_device_->get_device_ids(),
        ElementsAre(mesh_2x2_device_ids[0], mesh_2x2_device_ids[1], mesh_2x2_device_ids[2], mesh_2x2_device_ids[3]));
}

INSTANTIATE_TEST_SUITE_P(
    AllMeshOffsets,
    MeshDevice2x2WithOffsetReshapeTest,
    testing::Values(MeshCoordinate(0, 0), MeshCoordinate(0, 1), MeshCoordinate(0, 2)));

}  // namespace
}  // namespace tt::tt_metal::distributed
