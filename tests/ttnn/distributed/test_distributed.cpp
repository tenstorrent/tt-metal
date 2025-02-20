// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/mesh_coord.hpp>

#include <ttnn/core.hpp>
#include <ttnn/distributed/api.hpp>

namespace ttnn::distributed::test {

using ::tt::tt_metal::distributed::MeshContainer;

class DistributedTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(DistributedTest, TestSystemMeshTearDownWithoutClose) {
    auto& sys = SystemMesh::instance();
    auto mesh = ttnn::distributed::open_mesh_device(
        /*mesh_shape=*/{2, 4},
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,
        tt::tt_metal::DispatchCoreType::WORKER);

    const auto system_shape = sys.get_shape();
    ASSERT_EQ(system_shape.dims(), 2);
    EXPECT_EQ(system_shape[0], 2);
    EXPECT_EQ(system_shape[1], 4);
}

TEST_F(DistributedTest, TestMemoryAllocationStatistics) {
    auto mesh = ttnn::distributed::open_mesh_device(
        {2, 4}, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER);
    auto stats = mesh->allocator()->get_statistics(tt::tt_metal::BufferType::DRAM);
    for (auto* device : mesh->get_devices()) {
        auto device_stats = device->allocator()->get_statistics(tt::tt_metal::BufferType::DRAM);
        EXPECT_EQ(stats.total_allocatable_size_bytes, device_stats.total_allocatable_size_bytes);
    }
}

TEST_F(DistributedTest, TestNumDramChannels) {
    auto mesh = ttnn::distributed::open_mesh_device(
        {2, 4}, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER);
    EXPECT_EQ(mesh->num_dram_channels(), 96); // 8 devices * 12 channels
}

TEST_F(DistributedTest, ViewIs2D) {
    auto mesh = ttnn::distributed::open_mesh_device(
        {2, 4}, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER);
    std::vector<IDevice*> devices = mesh->get_devices();

    MeshContainer<IDevice*> container_1d(SimpleMeshShape(8), devices);
    MeshDeviceView view_1d(container_1d);
    EXPECT_FALSE(view_1d.is_mesh_2d());

    MeshContainer<IDevice*> container_2d(SimpleMeshShape(2, 4), devices);
    MeshDeviceView view_2d(container_2d);
    EXPECT_TRUE(view_2d.is_mesh_2d());

    MeshContainer<IDevice*> container_3d(SimpleMeshShape(2, 2, 2), devices);
    MeshDeviceView view_3d(container_3d);
    EXPECT_FALSE(view_3d.is_mesh_2d());
}

}  // namespace ttnn::distributed::test
