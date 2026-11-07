// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <string>
#include <stdexcept>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/info.hpp>
#include <tt-metalium/experimental/mesh_device.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed {
namespace {

namespace info = experimental::info;
using experimental::mesh_device::get_info;
using experimental::mesh_device::get_info_per_device;

using MeshDeviceInfoTest = GenericMeshDeviceFixture;
using MeshDeviceInfo1x1Test = MeshDevice1x1Fixture;

TEST_F(MeshDeviceInfoTest, UniformValuesMatchHal) {
    const auto mesh = get_mesh_device();

    EXPECT_EQ(get_info<info::l1_alignment>(*mesh), hal::get_l1_alignment());
    EXPECT_EQ(get_info<info::dram_alignment>(*mesh), hal::get_dram_alignment());
    EXPECT_EQ(get_info<info::architecture>(*mesh), hal::get_arch());
    EXPECT_EQ(get_info<info::architecture_name>(*mesh), hal::get_arch_name());
}

TEST_F(MeshDeviceInfoTest, PerCoordinateMatchesUniform) {
    const auto mesh = get_mesh_device();
    const uint32_t l1_alignment = get_info<info::l1_alignment>(*mesh);
    const tt::ARCH arch = get_info<info::architecture>(*mesh);

    for (const MeshCoordinate& coord : MeshCoordinateRange(mesh->shape())) {
        EXPECT_EQ(get_info<info::l1_alignment>(*mesh, coord), l1_alignment) << coord;
        EXPECT_EQ(get_info<info::architecture>(*mesh, coord), arch) << coord;
    }
}

TEST_F(MeshDeviceInfoTest, PerDeviceContainerCoversTheMesh) {
    const auto mesh = get_mesh_device();
    const uint32_t dram_alignment = get_info<info::dram_alignment>(*mesh);

    const auto per_device = get_info_per_device<info::dram_alignment>(*mesh);
    EXPECT_EQ(per_device.shape(), mesh->shape());
    for (const MeshCoordinate& coord : MeshCoordinateRange(mesh->shape())) {
        ASSERT_TRUE(per_device.is_local(coord)) << coord;
        EXPECT_EQ(per_device.at(coord).value(), dram_alignment) << coord;
    }
}

TEST_F(MeshDeviceInfo1x1Test, OutOfBoundsCoordinateThrows) {
    const auto mesh = get_mesh_device();
    ASSERT_EQ(mesh->shape(), MeshShape(1, 1));

    EXPECT_NO_THROW(get_info<info::l1_alignment>(*mesh, MeshCoordinate(0, 0)));
    EXPECT_THAT(
        [&] { get_info<info::l1_alignment>(*mesh, MeshCoordinate(1, 0)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Cannot query l1_alignment at")));
    EXPECT_THAT(
        [&] { get_info<info::dram_alignment>(*mesh, MeshCoordinate(0, 1)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Cannot query dram_alignment at")));
    EXPECT_THAT(
        [&] { get_info<info::architecture>(*mesh, MeshCoordinate(1, 0)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Cannot query architecture at")));
    EXPECT_THAT(
        [&] { get_info<info::architecture_name>(*mesh, MeshCoordinate(1, 0)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Cannot query architecture_name at")));
}

}  // namespace
}  // namespace tt::tt_metal::distributed
