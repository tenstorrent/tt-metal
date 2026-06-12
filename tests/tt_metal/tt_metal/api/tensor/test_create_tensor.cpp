// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <tuple>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/shape.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "common_tensor_test_utils.hpp"

namespace tt::tt_metal {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// =============================================================================
// CreateTensorTest: host->device->host byte round-trip for a tiled tensor.
// =============================================================================

struct CreateTensorParams {
    Shape shape;
};

class CreateTensorTest : public MeshDevice1x1Fixture, public ::testing::WithParamInterface<CreateTensorParams> {};

TEST_P(CreateTensorTest, Tile) {
    const CreateTensorParams& params = GetParam();
    const Shape& input_shape = params.shape;

    auto mem_cfg = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    constexpr DataType dtype = DataType::BFLOAT16;

    constexpr uint32_t datum_size_bytes = 2;

    TensorSpec tensor_spec(input_shape, TensorLayout(dtype, PageConfig(Layout::TILE), mem_cfg));
    ASSERT_EQ(input_shape.volume() * datum_size_bytes, tensor_spec.compute_packed_buffer_size_bytes());

    std::vector<bfloat16> host_data(input_shape.volume(), bfloat16(1.0f));

    auto host_tensor = HostTensor::from_vector(host_data, tensor_spec);
    auto device_tensor = enqueue_write_tensor(mesh_device_->mesh_command_queue(), host_tensor, *mesh_device_);
    auto readback_tensor = enqueue_read_tensor(mesh_device_->mesh_command_queue(), device_tensor);

    auto readback_data = readback_tensor.to_vector<bfloat16>();

    ASSERT_EQ(host_data.size(), readback_data.size());
    for (size_t i = 0; i < host_data.size(); i++) {
        EXPECT_EQ(host_data[i], readback_data[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    CreateTensorTestWithShape,
    CreateTensorTest,
    ::testing::Values(
        CreateTensorParams{.shape = Shape({1, 1, 32, 32})},
        CreateTensorParams{.shape = Shape({2, 1, 32, 32})},
        CreateTensorParams{.shape = Shape({0, 0, 0, 0})},
        CreateTensorParams{.shape = Shape({0, 1, 32, 32})},
        CreateTensorParams{.shape = Shape({0})}));

// =============================================================================
// EmptyTensorTest: round-trips a sweep of shape/dtype/layout/memory-config combos.
// =============================================================================

using CombinationInputParams = std::tuple<Shape, DataType, Layout, MemoryConfig>;

class EmptyTensorTest : public MeshDevice1x1Fixture, public ::testing::WithParamInterface<CombinationInputParams> {};

TEST_P(EmptyTensorTest, Combinations) {
    auto params = GetParam();
    auto shape = std::get<0>(params);
    auto dtype = std::get<1>(params);
    auto layout = std::get<2>(params);
    auto memory_config = std::get<3>(params);
    log_info(
        tt::LogTest,
        "Running test with shape={}, dtype={}, layout={}, memory_config={}",
        shape,
        dtype,
        layout,
        memory_config);

    if (layout == Layout::ROW_MAJOR && dtype == DataType::BFLOAT8_B) {
        GTEST_SKIP() << "Skipping test with ROW_MAJOR layout and BFLOAT8_B dtype!";
    }

    auto tensor_layout = TensorLayout::fromPaddedShape(
        dtype, PageConfig(layout), memory_config, /* logical */ shape, /* padded */ shape);

    auto tensor = MeshTensor::allocate_on_device(
        *mesh_device_, TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), memory_config)), TensorTopology());
    EXPECT_EQ(tensor.logical_shape(), shape);

    ::test_utils::test_tensor_on_device(shape, tensor_layout, *mesh_device_);
}

INSTANTIATE_TEST_SUITE_P(
    EmptyTensorTestWithShape,
    EmptyTensorTest,
    ::testing::Combine(
        ::testing::Values(
            Shape({}),
            Shape({0}),
            Shape({1}),
            Shape({1, 2}),
            Shape({1, 2, 3}),
            Shape({1, 2, 3, 4}),
            // Shape({0, 0, 0, 0}), fails with width sharded case
            Shape({1, 1, 1, 1}),
            // Shape({0, 1, 32, 32}), fails with width sharded case
            Shape({1, 1, 32, 32}),
            Shape({2, 1, 32, 32}),
            Shape({64, 1, 256, 1}),
            Shape({1, 1, 21120, 16}),
            Shape({1, 2, 3, 4, 5})),

        ::testing::Values(DataType::BFLOAT16, DataType::UINT32, DataType::FLOAT32, DataType::BFLOAT8_B),

        ::testing::Values(Layout::TILE, Layout::ROW_MAJOR),

        ::testing::Values(
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1},
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM})));

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace
}  // namespace tt::tt_metal
