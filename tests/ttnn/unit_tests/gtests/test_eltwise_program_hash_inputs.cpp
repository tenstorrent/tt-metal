// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Program-cache keys for the eltwise ops on compute_program_hash_inputs.
//
// ternary and binary_ng declare WHAT they key on (compute_program_hash_inputs); the framework derives both
// the 64-bit program hash and the exact canonical cache key from that tuple. Each op gets two checks:
//   *SharesKey      - a value that only reaches runtime args (a ttsl::runtime_only scalar, a tolerance, the
//                     row count of an interleaved operand) keeps both the hash and the canonical key, so one
//                     cached program still serves every value (no extra compiles).
//   *SeparatesKeys  - a value compiled into the program changes both, so it cannot be served a program
//                     built for another value (no stale hits, and no collision can hide the difference).
//
// Like test_stale_program_hash.cpp these stop at the key: no program is built or dispatched.

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp"
#include "ttnn/operations/eltwise/ternary/device/ternary_device_operation.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test::eltwise_program_hash_inputs {
namespace {

using ::tt::tt_metal::CoreCoord;
using ::tt::tt_metal::CoreRange;
using ::tt::tt_metal::CoreRangeSet;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorSpec;
using MeshDevice = ::tt::tt_metal::distributed::MeshDevice;

Tensor make_tile(MeshDevice* device, const ttnn::Shape& shape, DataType dtype = DataType::BFLOAT16) {
    return ttnn::create_device_tensor(
        TensorSpec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG)), device);
}

// The two halves of the key the program cache compares: the 64-bit hash, and on a hash hit the exact
// canonical encoding of the same inputs (what MeshDeviceOperationAdapter::compute_mesh_workload_canonical_key
// appends for an op on compute_program_hash_inputs).
struct Key {
    ttsl::hash::hash_t hash;
    std::string canonical;
};

template <typename Op>
Key key_of(const typename Op::operation_attributes_t& attrs, const typename Op::tensor_args_t& tensor_args) {
    static_assert(ttnn::device_operation::DeviceOperationWithProgramHashInputs<Op>);
    return {
        ttnn::device_operation::detail::compute_program_hash<Op>(attrs, tensor_args),
        ttsl::hash::canonical_key(Op::compute_program_hash_inputs(attrs, tensor_args))};
}

void expect_same_key(const Key& a, const Key& b) {
    EXPECT_EQ(a.hash, b.hash);
    EXPECT_EQ(a.canonical, b.canonical);
}

void expect_different_key(const Key& a, const Key& b) {
    EXPECT_NE(a.hash, b.hash);
    EXPECT_NE(a.canonical, b.canonical);
}

const CoreRangeSet kGrid{CoreRange{CoreCoord{0, 0}, CoreCoord{7, 7}}};

class EltwiseProgramHashInputsFixture : public TTNNFixtureWithSuiteDevice<EltwiseProgramHashInputsFixture> {};

// ---------------------------------------------------------------------------------------------------------------
// ternary: scalar_input_a/b are runtime_only; every other attribute and the shape-blind operand keys are keyed.
// ---------------------------------------------------------------------------------------------------------------

using TernaryOp = ::ttnn::operations::ternary::TernaryDeviceOperation;
namespace ternary = ::ttnn::operations::ternary;

TernaryOp::operation_attributes_t where_tts_attrs(float scalar, DataType dtype = DataType::BFLOAT16) {
    return {
        .ternary_op_type = ternary::TernaryOpType::WHERE,
        .ternary_variant = ternary::TernaryVariant::TTS,
        .broadcast_type = ternary::TernaryBroadcastType::NONE,
        .memory_config = ttnn::DRAM_MEMORY_CONFIG,
        .input_dtype = DataType::BFLOAT16,
        .worker_grid = kGrid,
        .dtype = dtype,
        .compute_kernel_config = std::nullopt,
        .sub_core_grids = std::nullopt,
        .scalar_input_a = std::nullopt,
        .scalar_input_b = ternary::ScalarVariant(scalar)};
}

TEST_F(EltwiseProgramHashInputsFixture, TernaryScalarSharesKey) {
    const auto predicate = make_tile(device_, ttnn::Shape{1, 1, 64, 64});
    const auto value_true = make_tile(device_, ttnn::Shape{1, 1, 64, 64});
    const TernaryOp::tensor_args_t args{predicate, value_true, std::nullopt, std::nullopt};
    expect_same_key(key_of<TernaryOp>(where_tts_attrs(0.0f), args), key_of<TernaryOp>(where_tts_attrs(-3.5f), args));
}

TEST_F(EltwiseProgramHashInputsFixture, TernaryInterleavedRowCountSharesKey) {
    // Same W, different (non-1) H: the interleaved program re-derives its per-core args on a hit.
    const auto p64 = make_tile(device_, ttnn::Shape{1, 1, 64, 64});
    const auto t64 = make_tile(device_, ttnn::Shape{1, 1, 64, 64});
    const auto p256 = make_tile(device_, ttnn::Shape{1, 1, 256, 64});
    const auto t256 = make_tile(device_, ttnn::Shape{1, 1, 256, 64});
    expect_same_key(
        key_of<TernaryOp>(where_tts_attrs(1.0f), {p64, t64, std::nullopt, std::nullopt}),
        key_of<TernaryOp>(where_tts_attrs(1.0f), {p256, t256, std::nullopt, std::nullopt}));
}

TEST_F(EltwiseProgramHashInputsFixture, TernaryCompiledInValuesSeparateKeys) {
    const auto predicate = make_tile(device_, ttnn::Shape{1, 1, 64, 64});
    const auto value_true = make_tile(device_, ttnn::Shape{1, 1, 64, 64});
    const TernaryOp::tensor_args_t args{predicate, value_true, std::nullopt, std::nullopt};
    const auto base = key_of<TernaryOp>(where_tts_attrs(1.0f), args);

    // Output dtype.
    expect_different_key(base, key_of<TernaryOp>(where_tts_attrs(1.0f, DataType::FLOAT32), args));

    // Operand width (the broadcast key keeps W).
    const auto p_wide = make_tile(device_, ttnn::Shape{1, 1, 64, 128});
    const auto t_wide = make_tile(device_, ttnn::Shape{1, 1, 64, 128});
    expect_different_key(base, key_of<TernaryOp>(where_tts_attrs(1.0f), {p_wide, t_wide, std::nullopt, std::nullopt}));

    // Operand dtype.
    const auto t_fp32 = make_tile(device_, ttnn::Shape{1, 1, 64, 64}, DataType::FLOAT32);
    expect_different_key(
        base, key_of<TernaryOp>(where_tts_attrs(1.0f), {predicate, t_fp32, std::nullopt, std::nullopt}));
}

// ---------------------------------------------------------------------------------------------------------------
// binary_ng: scalar / rtol / atol are runtime args; the shape-blind operand keys and the listed attributes are keyed.
// ---------------------------------------------------------------------------------------------------------------

using BinaryNgOp = ::ttnn::operations::binary_ng::BinaryNgDeviceOperation;
using ::ttnn::operations::binary_ng::BinaryOpType;

BinaryNgOp::operation_attributes_t binary_attrs(BinaryOpType op, std::optional<float> scalar, float rtol = 0.0f) {
    return {
        .binary_op_type = op,
        .scalar = scalar.has_value() ? std::optional<ttnn::operations::unary::ScalarVariant>{*scalar} : std::nullopt,
        .memory_config = ttnn::DRAM_MEMORY_CONFIG,
        .input_dtype = DataType::BFLOAT16,
        .dtype = DataType::BFLOAT16,
        .worker_grid = kGrid,
        .rtol = rtol};
}

TEST_F(EltwiseProgramHashInputsFixture, BinaryNgScalarAndToleranceShareKey) {
    const auto a = make_tile(device_, ttnn::Shape{1, 1, 64, 64});
    const BinaryNgOp::tensor_args_t scalar_args{a, std::nullopt, std::nullopt};
    expect_same_key(
        key_of<BinaryNgOp>(binary_attrs(BinaryOpType::ADD, 2.0f), scalar_args),
        key_of<BinaryNgOp>(binary_attrs(BinaryOpType::ADD, -7.25f), scalar_args));

    const auto b = make_tile(device_, ttnn::Shape{1, 1, 64, 64});
    const BinaryNgOp::tensor_args_t tensor_args{a, b, std::nullopt};
    expect_same_key(
        key_of<BinaryNgOp>(binary_attrs(BinaryOpType::ISCLOSE, std::nullopt, 1e-5f), tensor_args),
        key_of<BinaryNgOp>(binary_attrs(BinaryOpType::ISCLOSE, std::nullopt, 1e-2f), tensor_args));
}

TEST_F(EltwiseProgramHashInputsFixture, BinaryNgInterleavedShapeSharesKey) {
    const auto a_small = make_tile(device_, ttnn::Shape{1, 1, 64, 64});
    const auto b_small = make_tile(device_, ttnn::Shape{1, 1, 64, 64});
    const auto a_big = make_tile(device_, ttnn::Shape{2, 3, 128, 256});
    const auto b_big = make_tile(device_, ttnn::Shape{2, 3, 128, 256});
    expect_same_key(
        key_of<BinaryNgOp>(binary_attrs(BinaryOpType::ADD, std::nullopt), {a_small, b_small, std::nullopt}),
        key_of<BinaryNgOp>(binary_attrs(BinaryOpType::ADD, std::nullopt), {a_big, b_big, std::nullopt}));
}

TEST_F(EltwiseProgramHashInputsFixture, BinaryNgCompiledInValuesSeparateKeys) {
    const auto a = make_tile(device_, ttnn::Shape{1, 1, 64, 64});
    const auto b = make_tile(device_, ttnn::Shape{1, 1, 64, 64});
    const BinaryNgOp::tensor_args_t args{a, b, std::nullopt};
    const auto base = key_of<BinaryNgOp>(binary_attrs(BinaryOpType::ADD, std::nullopt), args);

    // Op type.
    expect_different_key(base, key_of<BinaryNgOp>(binary_attrs(BinaryOpType::MUL, std::nullopt), args));

    // Operand dtype (was folded into a lossy 64-bit to_hash leaf; now part of the exact key).
    const auto b_fp32 = make_tile(device_, ttnn::Shape{1, 1, 64, 64}, DataType::FLOAT32);
    expect_different_key(
        base, key_of<BinaryNgOp>(binary_attrs(BinaryOpType::ADD, std::nullopt), {a, b_fp32, std::nullopt}));

    // Tensor-tensor vs tensor-scalar.
    expect_different_key(
        base, key_of<BinaryNgOp>(binary_attrs(BinaryOpType::ADD, 1.0f), {a, std::nullopt, std::nullopt}));
}

}  // namespace
}  // namespace ttnn::test::eltwise_program_hash_inputs
