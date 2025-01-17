// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/system_mesh.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

TEST_F(T3000MultiDeviceFixture, SimpleMeshDeviceTest) {
    EXPECT_EQ(mesh_device_->num_devices(), 8);
    EXPECT_EQ(mesh_device_->num_rows(), 2);
    EXPECT_EQ(mesh_device_->num_cols(), 4);
}

TEST(MeshDeviceSuite, Test1x1SystemMeshInitialize) {
    auto& sys = tt::tt_metal::distributed::SystemMesh::instance();

    auto config = tt::tt_metal::distributed::MeshDeviceConfig{.mesh_shape = MeshShape(1, 1)};

    EXPECT_NO_THROW({
        auto mesh = tt::tt_metal::distributed::MeshDevice::create(
            config, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER);
        mesh->close();
    });
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
