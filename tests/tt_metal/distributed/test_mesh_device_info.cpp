// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <string>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/info.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed {
namespace {

using MeshDeviceInfoTest = GenericMeshDeviceFixture;
using MeshDeviceInfo1x1Test = MeshDevice1x1Fixture;

TEST_F(MeshDeviceInfoTest, UniformValuesMatchHal) {
    const auto mesh = get_mesh_device();

    EXPECT_EQ(mesh->get_info<info::l1_alignment>(), hal::get_l1_alignment());
    EXPECT_EQ(mesh->get_info<info::dram_alignment>(), hal::get_dram_alignment());
    EXPECT_EQ(mesh->get_info<info::architecture>(), hal::get_arch());
    EXPECT_EQ(mesh->get_info<info::architecture_name>(), hal::get_arch_name());
}

TEST_F(MeshDeviceInfoTest, PerCoordinateMatchesUniform) {
    const auto mesh = get_mesh_device();
    const uint32_t l1_alignment = mesh->get_info<info::l1_alignment>();
    const tt::ARCH arch = mesh->get_info<info::architecture>();

    for (const MeshCoordinate& coord : MeshCoordinateRange(mesh->shape())) {
        EXPECT_EQ(mesh->get_info<info::l1_alignment>(coord), l1_alignment) << coord;
        EXPECT_EQ(mesh->get_info<info::architecture>(coord), arch) << coord;
    }
}

TEST_F(MeshDeviceInfoTest, PerDeviceContainerCoversTheMesh) {
    const auto mesh = get_mesh_device();
    const uint32_t dram_alignment = mesh->get_info<info::dram_alignment>();

    const auto per_device = mesh->get_info_per_device<info::dram_alignment>();
    EXPECT_EQ(per_device.shape(), mesh->shape());
    for (const MeshCoordinate& coord : MeshCoordinateRange(mesh->shape())) {
        ASSERT_TRUE(per_device.is_local(coord)) << coord;
        EXPECT_EQ(per_device.at(coord).value(), dram_alignment) << coord;
    }
}

TEST_F(MeshDeviceInfo1x1Test, OutOfBoundsCoordinateThrows) {
    const auto mesh = get_mesh_device();
    ASSERT_EQ(mesh->shape(), MeshShape(1, 1));

    EXPECT_NO_THROW(mesh->get_info<info::l1_alignment>(MeshCoordinate(0, 0)));
    EXPECT_ANY_THROW(mesh->get_info<info::l1_alignment>(MeshCoordinate(1, 0)));
    EXPECT_ANY_THROW(mesh->get_info<info::l1_alignment>(MeshCoordinate(0, 1)));
}

}  // namespace
}  // namespace tt::tt_metal::distributed
