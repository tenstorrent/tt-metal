// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <ttnn/core.hpp>
#include <ttnn/distributed/api.hpp>

namespace ttnn::distributed::test {

class DistributedTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(DistributedTest, TestSystemMeshTearDownWithoutClose) {
    auto& sys = SystemMesh::instance();
    auto mesh = ttnn::distributed::open_mesh_device(
        {2, 4}, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER);

    auto [rows, cols] = sys.get_shape();
    EXPECT_GT(rows, 0);
    EXPECT_GT(cols, 0);
}

TEST_F(DistributedTest, TestMemoryAllocationStatistics) {
    auto mesh = ttnn::distributed::open_mesh_device(
        {2, 4}, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER);
    auto stats = mesh->get_memory_allocation_statistics(tt::tt_metal::BufferType::DRAM);
    for (auto* device : mesh->get_devices()) {
        auto device_stats = device->get_memory_allocation_statistics(tt::tt_metal::BufferType::DRAM);
        EXPECT_EQ(stats.total_allocatable_size_bytes, device_stats.total_allocatable_size_bytes);
    }
}

TEST_F(DistributedTest, TestNumDramChannels) {
    auto mesh = ttnn::distributed::open_mesh_device(
        {2, 4}, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER);
    EXPECT_EQ(mesh->num_dram_channels(), 96); // 8 devices * 12 channels
}

}  // namespace ttnn::distributed::test
