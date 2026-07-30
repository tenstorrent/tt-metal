// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>

#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/operations/experimental/quasar/conv2d/device/conv2d_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

// These ops had custom program hashes that keyed on less than the framework's reflection does. The
// hashes are gone; these pin what the key must still discriminate, so a narrowing regression fails here.
namespace ttnn {
namespace {

using ::ttnn::device_operation::detail::compute_program_hash;

Tensor host_tensor(const ttnn::Shape& logical_shape, Layout layout = Layout::TILE) {
    const tt::tt_metal::TensorSpec spec(
        logical_shape, tt::tt_metal::TensorLayout(DataType::FLOAT32, layout, MemoryConfig{}));
    return Tensor::from_vector(std::vector<float>(logical_shape.volume()), spec);
}

using ConvOp = ttnn::prim::qsr::Conv2dDeviceOperation;

ttnn::prim::Conv2dInputs conv_inputs() {
    return ttnn::prim::Conv2dInputs{
        .a = host_tensor(ttnn::Shape{1, 1, 64, 32}), .b = host_tensor(ttnn::Shape{1, 1, 32, 32}), .bias = std::nullopt};
}

// full_inner_dim selects slice_inner_dim in the sharded factory, so it changes the program. The removed
// custom hash left it out: these two convs shared one cache entry, and the second reused the first's
// program. Reflection over Conv2dParams keys it.
TEST(QuasarKeyingTest, Conv2dKeysFullInnerDim) {
    ttnn::prim::Conv2dParams params{};
    auto other = params;
    other.full_inner_dim = !params.full_inner_dim;

    const auto inputs = conv_inputs();
    EXPECT_NE(compute_program_hash<ConvOp>(params, inputs), compute_program_hash<ConvOp>(other, inputs));
}

TEST(QuasarKeyingTest, Conv2dKeepsKeyingTheOldHashFields) {
    const ttnn::prim::Conv2dParams params{};
    const auto inputs = conv_inputs();
    EXPECT_EQ(compute_program_hash<ConvOp>(params, inputs), compute_program_hash<ConvOp>(params, inputs));

    // A sample of the fields the removed hash listed; each must still split the key.
    auto changed_channels = params;
    changed_channels.output_channels = params.output_channels + 1;
    EXPECT_NE(compute_program_hash<ConvOp>(params, inputs), compute_program_hash<ConvOp>(changed_channels, inputs));

    auto changed_bias = params;
    changed_bias.has_bias = !params.has_bias;
    EXPECT_NE(compute_program_hash<ConvOp>(params, inputs), compute_program_hash<ConvOp>(changed_bias, inputs));

    auto changed_dtype = params;
    changed_dtype.dtype = DataType::BFLOAT16;
    EXPECT_NE(compute_program_hash<ConvOp>(params, inputs), compute_program_hash<ConvOp>(changed_dtype, inputs));

    // Tensor args are keyed too — the removed hash folded them in via the default reflection hash.
    ttnn::prim::Conv2dInputs other_inputs = conv_inputs();
    other_inputs.a = host_tensor(ttnn::Shape{1, 1, 128, 32});
    EXPECT_NE(compute_program_hash<ConvOp>(params, inputs), compute_program_hash<ConvOp>(params, other_inputs));
}

using SliceOp = ttnn::prim::qsr::SliceDeviceOperation;

ttnn::prim::qsr::SliceParams slice_params() {
    return ttnn::prim::qsr::SliceParams{
        .slice_start = ttnn::Shape{0, 0, 0, 0},
        .slice_end = ttnn::Shape{1, 1, 32, 32},
        .step = ttnn::Shape{1, 1, 1, 1},
        .output_mem_config = MemoryConfig{}};
}

// The removed hash listed slice params, the input spec (rank/logical/padded/layout/dtype/memory_config),
// the derived output spec and the selected factory index. padded_shape is computed from logical_shape +
// tensor_layout, and both derived values are pure functions of these same inputs, so reflection covers
// them — pinned here rather than argued.
TEST(QuasarKeyingTest, SliceKeysParamsAndInputSpec) {
    const auto params = slice_params();
    const ttnn::prim::qsr::SliceInputs inputs{.input = host_tensor(ttnn::Shape{1, 1, 64, 32})};
    EXPECT_EQ(compute_program_hash<SliceOp>(params, inputs), compute_program_hash<SliceOp>(params, inputs));

    auto moved_start = params;
    moved_start.slice_start = ttnn::Shape{0, 0, 32, 0};
    EXPECT_NE(compute_program_hash<SliceOp>(params, inputs), compute_program_hash<SliceOp>(moved_start, inputs));

    auto strided = params;
    strided.step = ttnn::Shape{1, 1, 2, 1};
    EXPECT_NE(compute_program_hash<SliceOp>(params, inputs), compute_program_hash<SliceOp>(strided, inputs));

    // Input spec: logical shape and layout both feed the factory choice and the work split.
    const ttnn::prim::qsr::SliceInputs taller{.input = host_tensor(ttnn::Shape{1, 1, 128, 32})};
    EXPECT_NE(compute_program_hash<SliceOp>(params, inputs), compute_program_hash<SliceOp>(params, taller));

    const ttnn::prim::qsr::SliceInputs row_major{.input = host_tensor(ttnn::Shape{1, 1, 64, 32}, Layout::ROW_MAJOR)};
    EXPECT_NE(compute_program_hash<SliceOp>(params, inputs), compute_program_hash<SliceOp>(params, row_major));
}

// start_tensor presence was keyed explicitly by the removed hash; end_tensor and preallocated_output
// were not keyed at all. Reflection keys the shape of tensor_args, so all three split the key now.
TEST(QuasarKeyingTest, SliceKeysOptionalTensorArgs) {
    const auto params = slice_params();
    const Tensor tensor = host_tensor(ttnn::Shape{1, 1, 64, 32});
    const ttnn::prim::qsr::SliceInputs bare{.input = tensor};

    ttnn::prim::qsr::SliceInputs with_start{.input = tensor};
    with_start.start_tensor = tensor;
    EXPECT_NE(compute_program_hash<SliceOp>(params, bare), compute_program_hash<SliceOp>(params, with_start));

    ttnn::prim::qsr::SliceInputs with_end{.input = tensor};
    with_end.end_tensor = tensor;
    EXPECT_NE(compute_program_hash<SliceOp>(params, bare), compute_program_hash<SliceOp>(params, with_end));

    // Same tensor in a different optional slot must not collide.
    EXPECT_NE(compute_program_hash<SliceOp>(params, with_start), compute_program_hash<SliceOp>(params, with_end));
}

}  // namespace
}  // namespace ttnn
