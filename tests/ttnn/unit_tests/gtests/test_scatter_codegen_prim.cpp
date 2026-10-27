// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The codegen scatter prim takes an optional preallocated output that no public entry exposes, so
// its contract on that argument is only reachable from C++: the destination must be the tensor the
// op would create for itself (standard padding included) and must not be one of the operands the
// kernels still read.

#include <gtest/gtest.h>

#include <cstdint>
#include <exception>
#include <optional>
#include <vector>

#include "ttnn/device.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/scatter/codegen/scatter_codegen_device_operation.hpp"
#include "ttnn/operations/data_movement/scatter/codegen/scatter_codegen_supported.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::data_movement::scatter::test {

class ScatterCodegenPrimFixture : public TTNNFixtureWithSuiteDevice<ScatterCodegenPrimFixture> {};

namespace {

struct Operands {
    Tensor input;
    Tensor index;
    Tensor src;
};

// bf16 TILE operands with the scatter axis last and a 32x32 index unique along it, the shape every
// TILE factory serves. `width` is the input's last extent; index and src share the input's rows.
Operands make_operands(tt::tt_metal::distributed::MeshDevice* device, uint32_t width) {
    const ttnn::Shape input_shape({1, 1, 32, width});
    const ttnn::Shape index_shape({1, 1, 32, 32});
    std::vector<int32_t> index(32 * 32);
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            index[r * 32 + c] = static_cast<int32_t>((c * (width / 32) + r) % width);
        }
    }
    const tt::tt_metal::TensorSpec index_spec(
        index_shape,
        tt::tt_metal::TensorLayout(
            DataType::INT32, tt::tt_metal::PageConfig(Layout::TILE), tt::tt_metal::MemoryConfig{}));
    return {
        ttnn::ones(input_shape, DataType::BFLOAT16, Layout::TILE, *device),
        Tensor::from_vector(index, index_spec, device),
        ttnn::zeros(index_shape, DataType::BFLOAT16, Layout::TILE, *device),
    };
}

Tensor run(const Operands& ops, const std::optional<Tensor>& output) {
    return ttnn::prim::scatter_codegen(
        ops.input, ops.index, ops.src, kReductionNone, ops.input.memory_config(), output, std::nullopt);
}

}  // namespace

TEST_F(ScatterCodegenPrimFixture, AcceptsTheDestinationItWouldCreateItself) {
    const Operands ops = make_operands(device_, 64);
    Tensor destination = ttnn::create_device_tensor(ops.input.tensor_spec(), device_);
    const Tensor out = run(ops, destination);
    EXPECT_EQ(out.buffer(), destination.buffer());
}

TEST_F(ScatterCodegenPrimFixture, RejectsADestinationPaddedPastTheStandardGeometry) {
    const Operands ops = make_operands(device_, 64);
    // Same logical shape as the input, but tilized with twice the padded rows: the TILE writers
    // compute page ids from the input's tile-rounded geometry, so these pages would not line up.
    const Tensor host = ttnn::zeros(ttnn::Shape({1, 1, 32, 64}), DataType::BFLOAT16, Layout::ROW_MAJOR, *device_);
    const Tensor padded = ttnn::tilize_with_val_padding(host, ttnn::Shape({1, 1, 64, 64}), ttnn::PadValue{0.0f});
    ASSERT_EQ(padded.logical_shape(), ops.input.logical_shape());
    ASSERT_EQ(padded.padded_shape(), ttnn::Shape({1, 1, 64, 64}));
    EXPECT_FALSE(supported_execution_controls(ops.input, ops.input.memory_config(), padded));
    EXPECT_THROW(run(ops, padded), std::exception);
}

TEST_F(ScatterCodegenPrimFixture, RejectsADestinationThatAliasesSrc) {
    // With src the input's own shape, src is a spec-exact destination; aliasing it would let the
    // writers overwrite source values the streaming readers have not consumed yet.
    const Operands ops = make_operands(device_, 32);
    ASSERT_EQ(ops.src.tensor_spec(), ops.input.tensor_spec());
    EXPECT_THROW(run(ops, ops.src), std::exception);
    EXPECT_NO_THROW(run(ops, std::nullopt));
}

}  // namespace ttnn::operations::data_movement::scatter::test
