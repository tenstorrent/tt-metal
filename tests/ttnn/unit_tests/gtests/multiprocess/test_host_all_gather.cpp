// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>
#include <ttnn/distributed/api.hpp>
#include <ttnn/distributed/host_ccl.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace ttnn::distributed {
namespace {

using ::testing::FloatEq;
using ::testing::Pointwise;
using ::tt::tt_fabric::MeshHostRankId;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::Tensor;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorSpec;

using BigMeshDualRankTest2x4 = tt::tt_metal::MeshDevice2x4Fixture;

int count_local_buffers(const Tensor& tensor) {
    int count = 0;
    tensor.host_storage().buffer().apply([&count](const auto&) { count++; });
    return count;
}

TEST_F(BigMeshDualRankTest2x4, HostAllGather) {
    constexpr int kNumDevices = 8;
    ASSERT_EQ(mesh_device_->num_devices(), kNumDevices);

    std::vector<float> test_data;
    for (int i = 0; i < kNumDevices; i++) {
        test_data.insert(test_data.end(), {i * 1.F, i * 2.F, i * 3.F});
    }
    Tensor input_tensor = Tensor::from_vector(
        test_data,
        TensorSpec(
            ttnn::Shape{1, kNumDevices, 3, 1}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{})));

    auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 1);
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper);

    ASSERT_EQ(count_local_buffers(sharded_tensor), kNumDevices / 2);

    // Perform all-gather on host and validate the data at each host.
    auto all_gather_tensor = host_ccl::all_gather(sharded_tensor);
    EXPECT_EQ(all_gather_tensor.storage_type(), tt::tt_metal::StorageType::HOST);
    EXPECT_EQ(count_local_buffers(all_gather_tensor), kNumDevices);

    auto composer = concat_mesh_to_tensor_composer(*mesh_device_, /*dim=*/0);

    EXPECT_THAT(aggregate_tensor(all_gather_tensor, *composer).to_vector<float>(), Pointwise(FloatEq(), test_data));

    // Calling `all_gather` again should be a no-op.
    all_gather_tensor = host_ccl::all_gather(all_gather_tensor);
    EXPECT_EQ(all_gather_tensor.storage_type(), tt::tt_metal::StorageType::HOST);
    EXPECT_EQ(count_local_buffers(all_gather_tensor), kNumDevices);

    EXPECT_THAT(aggregate_tensor(all_gather_tensor, *composer).to_vector<float>(), Pointwise(FloatEq(), test_data));
}

}  // namespace
}  // namespace ttnn::distributed
