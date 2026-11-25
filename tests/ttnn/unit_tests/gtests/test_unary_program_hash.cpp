// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Cache-key tests for UnaryDeviceOperation::compute_program_hash.
//
// These stop at the hash generation intentionally. Both DFBs are sized from tile_size(DataFormat)
// in unary_program_factory.cpp, which assumes a fixed 32x32 tile. Dispatching a different tile size
// will write past the output buffer and returns wrong data.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <tt-metalium/vocabulary/core_coord.hpp>

#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_device_operation.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::unary::test {
namespace {

using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorSpec;
using ::tt::tt_metal::Tile;

const ttnn::Shape kShape{1, 1, 64, 64};

Tensor make_tensor(tt::tt_metal::distributed::MeshDevice* device, const std::array<uint32_t, 2>& tile) {
    return ttnn::create_device_tensor(
        TensorSpec(
            kShape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE, Tile(tile)), ttnn::DRAM_MEMORY_CONFIG)),
        device);
}

UnaryDeviceOperation::operation_attributes_t make_attributes(tt::tt_metal::distributed::MeshDevice* device) {
    const auto grid = device->compute_with_storage_grid_size();
    return UnaryDeviceOperation::operation_attributes_t{
        .op_chain = {EltwiseUnaryWithParam{UnaryOpType::RELU}},
        .output_dtype = DataType::BFLOAT16,
        .memory_config = ttnn::DRAM_MEMORY_CONFIG,
        .worker_grid = CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1})),
    };
}

class UnaryProgramHashFixture : public TTNNFixtureWithSuiteDevice<UnaryProgramHashFixture> {};

TEST_F(UnaryProgramHashFixture, SameTileSharesKey) {
    const auto attributes = make_attributes(device_);
    const auto first = make_tensor(device_, {32, 32});
    const auto second = make_tensor(device_, {32, 32});

    EXPECT_EQ(
        UnaryDeviceOperation::compute_program_hash(attributes, {first, std::nullopt}),
        UnaryDeviceOperation::compute_program_hash(attributes, {second, std::nullopt}));
}

TEST_F(UnaryProgramHashFixture, InputTileSeparatesKeys) {
    const auto attributes = make_attributes(device_);
    const auto tile_32x32 = make_tensor(device_, {32, 32});
    const auto tile_16x32 = make_tensor(device_, {16, 32});

    ASSERT_EQ(tile_32x32.padded_shape(), tile_16x32.padded_shape());
    EXPECT_NE(
        UnaryDeviceOperation::compute_program_hash(attributes, {tile_32x32, std::nullopt}),
        UnaryDeviceOperation::compute_program_hash(attributes, {tile_16x32, std::nullopt}));
}

TEST_F(UnaryProgramHashFixture, OutputTileSeparatesKeys) {
    const auto attributes = make_attributes(device_);
    const auto input = make_tensor(device_, {32, 32});
    const auto output_32x32 = make_tensor(device_, {32, 32});
    const auto output_16x32 = make_tensor(device_, {16, 32});

    EXPECT_NE(
        UnaryDeviceOperation::compute_program_hash(attributes, {input, output_32x32}),
        UnaryDeviceOperation::compute_program_hash(attributes, {input, output_16x32}));
}

constexpr std::array kRuntimeScalarOps = {
    UnaryOpType::FILL,           UnaryOpType::RELU_MAX,       UnaryOpType::RELU_MIN,       UnaryOpType::POWER,
    UnaryOpType::LEAKY_RELU,     UnaryOpType::HEAVISIDE,      UnaryOpType::BITWISE_XOR,    UnaryOpType::BITWISE_AND,
    UnaryOpType::BITWISE_OR,     UnaryOpType::RIGHT_SHIFT,    UnaryOpType::LEFT_SHIFT,     UnaryOpType::REMAINDER,
    UnaryOpType::FMOD,           UnaryOpType::RDIV,           UnaryOpType::RSUB,           UnaryOpType::RPOW,
    UnaryOpType::ADD_UNARY_SFPU, UnaryOpType::SUB_UNARY_SFPU, UnaryOpType::MUL_UNARY_SFPU, UnaryOpType::DIV_UNARY_SFPU,
    UnaryOpType::UNARY_NE,       UnaryOpType::UNARY_EQ,       UnaryOpType::UNARY_GT,       UnaryOpType::UNARY_LT,
    UnaryOpType::UNARY_GE,       UnaryOpType::UNARY_LE,       UnaryOpType::SOFTPLUS,       UnaryOpType::XIELU,
    UnaryOpType::PRELU_SFPU,     UnaryOpType::MAXIMUM,        UnaryOpType::MINIMUM,        UnaryOpType::CELU,
    UnaryOpType::SOFTSHRINK,     UnaryOpType::CLAMP_TSS,      UnaryOpType::HARDTANH,       UnaryOpType::THRESHOLD,
    UnaryOpType::SELU,
};

TEST(UnaryRuntimeScalars, ListedOpsAreTheOnesThatTakeThem) {
    for (const auto op_type : kRuntimeScalarOps) {
        EXPECT_TRUE(utils::takes_runtime_scalars(op_type)) << enchantum::to_string(op_type);
    }
    EXPECT_FALSE(utils::takes_runtime_scalars(UnaryOpType::ELU));
    EXPECT_FALSE(utils::takes_runtime_scalars(UnaryOpType::HARDSHRINK));
    EXPECT_FALSE(utils::takes_runtime_scalars(UnaryOpType::ROUND));
}

// The program hash keys an op by without_runtime_scalars. Two ops it maps to the same value must therefore generate
// the same code, or a cached program would run with code compiled for another op's parameters. Returns whether the
// code read any runtime scalars.
template <typename T>
bool expect_code_independent_of_runtime_scalars(UnaryOpType op_type, DataType dtype) {
    const auto with_values = [&](T value0, T value1) {
        // Param 1 of RDIV is its rounding mode, which stays compiled in.
        return EltwiseUnaryWithParam(op_type, std::vector<T>{value0, op_type == UnaryOpType::RDIV ? T{0} : value1});
    };
    const auto first = with_values(T{1}, T{3});
    const auto second = with_values(T{2}, T{5});
    const auto cache_key = [](const EltwiseUnaryWithParam& op) {
        const auto projected = utils::without_runtime_scalars(op);
        const auto params = projected.template get_params_if<T>();
        return std::vector<T>(params.begin(), params.end());
    };
    EXPECT_EQ(cache_key(first), cache_key(second)) << enchantum::to_string(op_type);

    const auto generate = [&](const EltwiseUnaryWithParam& op, std::vector<uint32_t>& values) {
        return utils::get_unary_chain_defines({op}, dtype, values);
    };
    std::vector<uint32_t> first_values;
    std::map<std::string, std::string> first_defines;
    try {
        first_defines = generate(first, first_values);
    } catch (const std::exception&) {
        return false;  // op_type does not support dtype or T params
    }
    std::vector<uint32_t> second_values;
    EXPECT_EQ(first_defines, generate(second, second_values))
        << enchantum::to_string(op_type) << " " << enchantum::to_string(dtype);
    EXPECT_EQ(first_values.empty(), second_values.empty())
        << enchantum::to_string(op_type) << " " << enchantum::to_string(dtype);
    return !first_values.empty();
}

TEST(UnaryRuntimeScalars, CodeIsIndependentOfRuntimeScalars) {
    for (const auto op_type : kRuntimeScalarOps) {
        bool read_runtime_scalars = false;
        for (const auto dtype : {DataType::BFLOAT16, DataType::FLOAT32, DataType::INT32, DataType::UINT32}) {
            read_runtime_scalars |= expect_code_independent_of_runtime_scalars<float>(op_type, dtype);
        }
        read_runtime_scalars |= expect_code_independent_of_runtime_scalars<int32_t>(op_type, DataType::INT32);
        read_runtime_scalars |= expect_code_independent_of_runtime_scalars<uint32_t>(op_type, DataType::UINT32);
        EXPECT_TRUE(read_runtime_scalars) << enchantum::to_string(op_type);
    }
}

}  // namespace
}  // namespace ttnn::operations::unary::test
