// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/device_tensor.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/shape.hpp>

namespace tt::tt_metal {

TEST(ExperimentalTensorTest, DefaultConstructedHostTensor) {
    // Default construct a HostTensor
    HostTensor tensor;

    // Shape-related
    const auto& logical_shape = tensor.logical_shape();
    EXPECT_EQ(logical_shape, Shape()) << "Default constructed HostTensor should have empty shape";
    // TODO: This should be 0, not 1 (Shape().volume())
    EXPECT_EQ(tensor.logical_volume(), Shape().volume())
        << "logical_volume() should return the volume of the empty shape";

    // TODO: padded shape should be empty?
    auto expected_padded_shape = Shape({32, 32});
    const auto& padded_shape = tensor.padded_shape();
    EXPECT_EQ(padded_shape, expected_padded_shape) << "Padded shape should be minimum 32x32";
    EXPECT_EQ(tensor.padded_volume(), expected_padded_shape.volume())
        << "padded_volume() should return the volume of the padded shape";

    auto strides = tensor.strides();
    EXPECT_TRUE(strides.empty()) << "Strides should be empty for empty shape";

    // Data, layout, memory config
    EXPECT_EQ(tensor.dtype(), DataType::INVALID) << "Default constructed HostTensor should have INVALID data type";
    EXPECT_EQ(tensor.layout(), Layout::INVALID) << "Default constructed HostTensor should have INVALID layout";

    // Sharding
    EXPECT_FALSE(tensor.is_sharded()) << "Default HostTensor is not sharded";
    EXPECT_FALSE(tensor.shard_spec().has_value()) << "Default HostTensor should have no shard_spec";
    EXPECT_FALSE(tensor.nd_shard_spec().has_value()) << "Default HostTensor should have no nd_shard_spec";

    // Storage & Element size
    EXPECT_THROW({ tensor.element_size(); }, std::exception) << "element_size() should throw";
    EXPECT_EQ(tensor.get_host_buffer(), HostBuffer()) << "get_host_buffer() should return an empty buffer";

    // Tensor topology
    EXPECT_NO_THROW(tensor.tensor_topology()) << "tensor_topology() should be accessible";

    // String conversion
    EXPECT_NO_THROW({ tensor.write_to_string(); }) << "write_to_string() should not throw";
}

}  // namespace tt::tt_metal
