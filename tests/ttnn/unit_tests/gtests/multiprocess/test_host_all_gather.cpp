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

using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::tt::tt_fabric::HostRankId;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::Tensor;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorSpec;

using BigMeshDualRankTest2x4 = tt::tt_metal::MeshDevice2x4Fixture;

TEST_F(BigMeshDualRankTest2x4, HostAllGather) {
    constexpr int num_devices = 8;
    ASSERT_EQ(mesh_device_->num_devices(), num_devices);

    std::vector<float> test_data;
    for (int i = 0; i < num_devices; i++) {
        test_data.insert(test_data.end(), {i * 1.F, i * 2.F, i * 3.F});
    }
    Tensor input_tensor = Tensor::from_vector(
        test_data,
        TensorSpec(
            ttnn::Shape{1, num_devices, 3, 1}, TensorLayout(DataType::FLOAT32, Layout::ROW_MAJOR, MemoryConfig{})));

    auto mapper = shard_tensor_to_mesh_mapper(*mesh_device_, 1);
    Tensor sharded_tensor = distribute_tensor(input_tensor, *mapper);

    std::vector<Tensor> device_tensors = get_device_tensors(sharded_tensor);
    ASSERT_EQ(device_tensors.size(), 4);

    // Perform all-gather on host and validate the data at each host.
    auto all_gather_tensor = host_ccl::all_gather(sharded_tensor);
    EXPECT_EQ(all_gather_tensor.storage_type(), tt::tt_metal::StorageType::HOST);
    EXPECT_THAT(get_device_tensors(all_gather_tensor), SizeIs(num_devices));

    auto composer = concat_mesh_to_tensor_composer(*mesh_device_, /*dim=*/0);
    Tensor concatenated_tensor = aggregate_tensor(all_gather_tensor, *composer);

    EXPECT_THAT(concatenated_tensor.to_vector<float>(), Pointwise(FloatEq(), test_data));
}

}  // namespace
}  // namespace ttnn::distributed
