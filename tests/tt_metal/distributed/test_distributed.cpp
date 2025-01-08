// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/distributed/distributed_fixture.hpp"

namespace tt::tt_metal::distributed::test {

TEST_F(MeshDevice_T3000, SimpleMeshDeviceTest) {
    EXPECT_EQ(mesh_device_->num_devices(), 8);
    EXPECT_EQ(mesh_device_->num_rows(), 2);
    EXPECT_EQ(mesh_device_->num_cols(), 4);
}

TEST(MeshDeviceSuite, Test1x1SystemMeshInitialize) {
    auto& sys = tt::tt_metal::distributed::SystemMesh::instance();

    auto config =
        tt::tt_metal::distributed::MeshDeviceConfig(MeshShape(1, 1), MeshOffset(0, 0), {}, MeshType::RowMajor);

    EXPECT_NO_THROW({
        auto mesh = tt::tt_metal::distributed::MeshDevice::create(
            config, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER);
        mesh->close();
    });
}

}  // namespace tt::tt_metal::distributed::test
