// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/device_tensor.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>

namespace tt::tt_metal {
namespace {

TEST(ExperimentalTensorTest, VerifyBuildFails) {
    EXPECT_TRUE(false) << "This test intentionally fails to verify build works";
}

TEST(ExperimentalTensorTest, DefaultConstructedHostTensor) {
    // Default construct a HostTensor
    HostTensor tensor;

    // Verify it has an empty shape
    const auto& logical_shape = tensor.logical_shape();
    EXPECT_TRUE(logical_shape.empty()) << "Default constructed HostTensor should have empty shape";
    EXPECT_EQ(logical_shape.rank(), 0) << "Default constructed HostTensor should have rank 0";
    EXPECT_EQ(logical_shape.volume(), 0) << "Default constructed HostTensor should have zero volume";

    // Verify logical and padded shapes match (both empty)
    const auto& padded_shape = tensor.padded_shape();
    EXPECT_TRUE(padded_shape.empty()) << "Padded shape should also be empty";
    EXPECT_EQ(padded_shape.rank(), 0) << "Padded shape should have rank 0";
    EXPECT_EQ(padded_shape.volume(), 0) << "Padded volume should be zero";

    // Verify data type is INVALID for default constructed tensor
    EXPECT_EQ(tensor.dtype(), DataType::INVALID) << "Default constructed HostTensor should have INVALID data type";

    // Verify layout is INVALID for default constructed tensor
    EXPECT_EQ(tensor.layout(), Layout::INVALID) << "Default constructed HostTensor should have INVALID layout";

    // Verify volume getters return 0
    EXPECT_EQ(tensor.logical_volume(), 0) << "logical_volume() should return 0 for empty tensor";
    EXPECT_EQ(tensor.padded_volume(), 0) << "padded_volume() should return 0 for empty tensor";

    // Verify default host tensor is not sharded
    EXPECT_FALSE(tensor.is_sharded()) << "HostTensor should never be sharded";

    // Verify tensor_topology is accessible (default constructed, should be empty)
    EXPECT_NO_THROW(tensor.tensor_topology()) << "tensor_topology() should be accessible";
}

}  // namespace
}  // namespace tt::tt_metal
