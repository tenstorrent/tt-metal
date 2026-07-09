// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/shape.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
#include <tt-metalium/experimental/tensor/tensor_types.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

struct Inputs {
    Shape shape;
    TensorLayout layout;
};

struct Expected {
    Shape padded_shape;
};

struct CreateTensorParams {
    Inputs inputs;
    Expected expected;
};

class CreateTensorWithLayoutTest : public GenericMeshDeviceFixture,
                                   public ::testing::WithParamInterface<CreateTensorParams> {};

TEST_P(CreateTensorWithLayoutTest, Tile) {
    const CreateTensorParams& params = GetParam();

    auto tensor = MeshTensor::allocate_on_device(
        *mesh_device_, TensorSpec(params.inputs.shape, params.inputs.layout), TensorTopology());
    EXPECT_EQ(tensor.padded_shape(), params.expected.padded_shape);
    EXPECT_EQ(tensor.logical_shape(), params.inputs.shape);
}

const MemoryConfig DefaultMemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt};

INSTANTIATE_TEST_SUITE_P(
    CreateTensorWithLayoutTestWithShape,
    CreateTensorWithLayoutTest,
    ::testing::Values(
        CreateTensorParams{
            Inputs{
                .shape = Shape({1, 1, 32, 32}),
                .layout = TensorLayout(DataType::BFLOAT16, Layout::TILE, DefaultMemoryConfig)},
            Expected{.padded_shape = Shape({1, 1, 32, 32})}},

        CreateTensorParams{
            Inputs{
                .shape = Shape({1, 1, 16, 10}),
                .layout = TensorLayout(DataType::BFLOAT16, Layout::TILE, DefaultMemoryConfig)},
            Expected{.padded_shape = Shape({1, 1, 32, 32})}},

        CreateTensorParams{
            Inputs{
                .shape = Shape({1, 1, 16, 10}),
                .layout = TensorLayout(DataType::BFLOAT16, Layout::ROW_MAJOR, DefaultMemoryConfig)},
            Expected{.padded_shape = Shape({1, 1, 16, 10})}}));

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace
}  // namespace tt::tt_metal
