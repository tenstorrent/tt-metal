// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/matmul_op.hpp"

#include <fmt/ranges.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tt_stl/reflection.hpp>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/core/xmath.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"

namespace {

// out = op(a) @ op(b) with op(a): [M, K], op(b): [K, N]. a is stored as [K, M] when transpose_a, b as [N, K] when
// transpose_b. batch == 0 keeps a and b rank 2; batch > 0 stores a as [batch, 1, ., .] and b as [1, 1, ., .], so
// backward has to flatten a's and dL/dout's batch into rows and reduce dB over it.
struct MatmulFrozenCase {
    std::string name;
    uint32_t M = 0;
    uint32_t K = 0;
    uint32_t N = 0;
    bool transpose_a = false;
    bool transpose_b = false;
    bool a_requires_grad = true;
    bool b_requires_grad = true;
    uint32_t batch = 0;
};

// Small integers keep every product and partial sum exact in bf16: |a|, |b| <= 2 and |out_grad| <= 1 bound each grad
// entry by 2 * max(batch * M, N) <= 192 < 256, so the device grads must equal the host reference bit for bit.
xt::xarray<float> make_small_ints(const std::vector<uint32_t>& shape, float bound, uint32_t seed) {
    return xt::round(ttml::test_utils::make_uniform_xarray<float>(shape, -bound, bound, seed));
}

// Copy of t with its leading dims flattened into rows: [..., rows, cols] -> [volume / cols, cols].
xt::xarray<float> flatten_rows(const xt::xarray<float>& t) {
    xt::xarray<float> flat = t;
    flat.reshape(std::vector<size_t>{t.size() / t.shape().back(), t.shape().back()});
    return flat;
}

// Host inputs plus the autograd graph of out = matmul_op(a, b), with out's grad set and backward not yet run.
struct MatmulFrozenGraph {
    xt::xarray<float> a;
    xt::xarray<float> b;
    xt::xarray<float> out_grad;
    ttml::autograd::TensorPtr t_a;
    ttml::autograd::TensorPtr t_b;
    ttml::autograd::TensorPtr out;
};

MatmulFrozenGraph build_graph(const MatmulFrozenCase& c) {
    auto* device = &ttml::autograd::ctx().get_device();
    MatmulFrozenGraph g;
    std::vector<uint32_t> a_shape = c.transpose_a ? std::vector{c.K, c.M} : std::vector{c.M, c.K};
    std::vector<uint32_t> b_shape = c.transpose_b ? std::vector{c.N, c.K} : std::vector{c.K, c.N};
    std::vector<uint32_t> out_grad_shape{c.M, c.N};
    if (c.batch > 0U) {
        a_shape.insert(a_shape.begin(), {c.batch, 1U});
        b_shape.insert(b_shape.begin(), {1U, 1U});
        out_grad_shape.insert(out_grad_shape.begin(), {c.batch, 1U});
    }
    g.a = make_small_ints(a_shape, /*bound=*/2.4F, /*seed=*/1U);
    g.b = make_small_ints(b_shape, /*bound=*/2.4F, /*seed=*/2U);
    g.out_grad = make_small_ints(out_grad_shape, /*bound=*/1.4F, /*seed=*/3U);
    g.t_a = ttml::autograd::create_tensor(ttml::core::from_xtensor(g.a, device), c.a_requires_grad);
    g.t_b = ttml::autograd::create_tensor(ttml::core::from_xtensor(g.b, device), c.b_requires_grad);
    g.out = ttml::ops::matmul_op(g.t_a, g.t_b, c.transpose_a, c.transpose_b);
    g.out->set_grad(ttml::core::from_xtensor(g.out_grad, device));
    return g;
}

}  // namespace

class MatmulOpTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

class MatmulOpFrozenInputTest : public MatmulOpTest, public ::testing::WithParamInterface<MatmulFrozenCase> {};

// ttnn's matmul rejects a batched transpose_a forward unless op(a)'s M fits in one tile, so M = 32 is the shape that
// would reach matmul_op's backward. There, flattening a [batch, K, M] a misaligns its rows with dL/dout's, so matmul_op
// must refuse it at the call site.
TEST_F(MatmulOpTest, BatchedTransposeARejectedInForward) {
    EXPECT_THAT(
        [] {
            build_graph(MatmulFrozenCase{
                .name = "BatchedTransposeA",
                .M = 32,
                .K = 64,
                .N = 32,
                .transpose_a = true,
                .b_requires_grad = false,
                .batch = 2});
        },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("does not support transpose_a with a batched a")));
}

// Backward reduces dB over a's flattened batch, so a per-batch b must be refused at the call site. With gradients
// disabled no backward node is recorded, so the same forward must run.
TEST_F(MatmulOpTest, BatchedBRejectedOnlyWhenRecordingGrad) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto a = ttml::autograd::create_tensor(
        ttml::core::from_xtensor(make_small_ints({2U, 1U, 32U, 64U}, /*bound=*/2.4F, /*seed=*/1U), device),
        /*requires_grad=*/true);
    const auto b = ttml::autograd::create_tensor(
        ttml::core::from_xtensor(make_small_ints({2U, 1U, 64U, 96U}, /*bound=*/2.4F, /*seed=*/2U), device),
        /*requires_grad=*/false);
    EXPECT_THAT(
        [&] { ttml::ops::matmul_op(a, b, /*transpose_a=*/false, /*transpose_b=*/false); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("does not support a batched b")));

    ttml::autograd::ctx().set_gradient_mode(ttml::autograd::GradMode::DISABLED);
    ttml::autograd::TensorPtr out;
    EXPECT_NO_THROW(out = ttml::ops::matmul_op(a, b, /*transpose_a=*/false, /*transpose_b=*/false));
    ttml::autograd::ctx().set_gradient_mode(ttml::autograd::GradMode::ENABLED);
    ASSERT_NE(out, nullptr);
    EXPECT_FALSE(out->get_node().has_value());
}

// add_grad drops a frozen tensor's grad, so the grads alone cannot show that a frozen side's matmul was skipped.
// backward runs once under a NORMAL-mode capture: the ops really execute, giving the grads to check, and the trace
// counts the matmuls launched.
TEST_P(MatmulOpFrozenInputTest, BackwardComputesOnlyTrainableGrads) {
    const MatmulFrozenCase c = GetParam();
    const MatmulFrozenGraph g = build_graph(c);

    // With both inputs frozen add_backward_node records no node, so matmul_op's grad function can never run.
    const bool any_trainable = c.a_requires_grad || c.b_requires_grad;
    ASSERT_EQ(g.out->get_node().has_value(), any_trainable) << c.name;
    if (!any_trainable) {
        return;
    }

    // The scoped capture ends even if backward throws, so a failing row cannot leave later tests capturing.
    ttnn::graph::ScopedGraphCapture capture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
    g.out->backward();
    const auto calls = ttnn::graph::extract_calltrace(capture.end_graph_capture());

    // One grad matmul per trainable input. The name comes from the op type, so renaming the op breaks the build rather
    // than silently counting zero.
    const std::string matmul_name{ttsl::get_type_name<ttnn::prim::MatmulDeviceOperation>()};
    const auto num_matmuls = std::ranges::count(calls, matmul_name);
    const auto expected_matmuls = static_cast<int>(c.a_requires_grad) + static_cast<int>(c.b_requires_grad);
    EXPECT_EQ(num_matmuls, expected_matmuls)
        << c.name << " backward ops: " << fmt::format("{}", fmt::join(calls, ", "));

    // With a's and dL/dout's batch flattened into rows, dL/d op(a) = out_grad @ op(b)^T and dL/d op(b) = op(a)^T @
    // out_grad, transposed back to the stored layout. With transpose_a only batch 1 is supported, so a flattens to
    // its single [K, M] matrix.
    const xt::xarray<float> a_2d = flatten_rows(g.a);
    const xt::xarray<float> b_2d = flatten_rows(g.b);
    const xt::xarray<float> out_grad_2d = flatten_rows(g.out_grad);
    const xt::xarray<float> op_a = c.transpose_a ? xt::transpose(a_2d) : a_2d;
    const xt::xarray<float> op_b = c.transpose_b ? xt::transpose(b_2d) : b_2d;
    const xt::xarray<float> d_op_a = xt::linalg::dot(out_grad_2d, xt::transpose(op_b));
    const xt::xarray<float> d_op_b = xt::linalg::dot(xt::transpose(op_a), out_grad_2d);

    if (c.a_requires_grad) {
        ASSERT_TRUE(g.t_a->is_grad_initialized()) << c.name;
        xt::xarray<float> expected = c.transpose_a ? xt::transpose(d_op_a) : d_op_a;
        expected.reshape(g.a.shape());
        EXPECT_TRUE(xt::all(xt::equal(ttml::core::to_xtensor(g.t_a->get_grad()), expected))) << c.name << " grad_a";
    }
    if (c.b_requires_grad) {
        ASSERT_TRUE(g.t_b->is_grad_initialized()) << c.name;
        xt::xarray<float> expected = c.transpose_b ? xt::transpose(d_op_b) : d_op_b;
        expected.reshape(g.b.shape());
        EXPECT_TRUE(xt::all(xt::equal(ttml::core::to_xtensor(g.t_b->get_grad()), expected))) << c.name << " grad_b";
    }
}

// transpose_a rows use a rectangular a (stored [K, M] = [64, 96]), so a wrong dL/dout reshape cannot pass by accident.
INSTANTIATE_TEST_SUITE_P(
    MatmulFrozenCases,
    MatmulOpFrozenInputTest,
    ::testing::Values(
        MatmulFrozenCase{.name = "BothTrainable", .M = 32, .K = 64, .N = 96},
        MatmulFrozenCase{
            .name = "BothFrozen", .M = 32, .K = 64, .N = 96, .a_requires_grad = false, .b_requires_grad = false},
        MatmulFrozenCase{.name = "FrozenB", .M = 32, .K = 64, .N = 96, .b_requires_grad = false},
        MatmulFrozenCase{
            .name = "FrozenB_TransposeA", .M = 96, .K = 64, .N = 32, .transpose_a = true, .b_requires_grad = false},
        MatmulFrozenCase{
            .name = "FrozenB_TransposeB", .M = 32, .K = 64, .N = 96, .transpose_b = true, .b_requires_grad = false},
        MatmulFrozenCase{
            .name = "FrozenB_TransposeBoth",
            .M = 96,
            .K = 64,
            .N = 32,
            .transpose_a = true,
            .transpose_b = true,
            .b_requires_grad = false},
        MatmulFrozenCase{.name = "FrozenA", .M = 32, .K = 64, .N = 96, .a_requires_grad = false},
        MatmulFrozenCase{
            .name = "FrozenA_TransposeA", .M = 96, .K = 64, .N = 32, .transpose_a = true, .a_requires_grad = false},
        MatmulFrozenCase{
            .name = "FrozenA_TransposeB", .M = 32, .K = 64, .N = 96, .transpose_b = true, .a_requires_grad = false},
        MatmulFrozenCase{
            .name = "FrozenA_TransposeBoth",
            .M = 96,
            .K = 64,
            .N = 32,
            .transpose_a = true,
            .transpose_b = true,
            .a_requires_grad = false},
        MatmulFrozenCase{.name = "Batched", .M = 32, .K = 64, .N = 96, .batch = 2},
        MatmulFrozenCase{.name = "Batched_TransposeB", .M = 32, .K = 64, .N = 96, .transpose_b = true, .batch = 2},
        // A batch of 1 flattens to the 2D case, so the batched transpose_a check must let it through.
        MatmulFrozenCase{.name = "BatchOne_TransposeA", .M = 96, .K = 64, .N = 32, .transpose_a = true, .batch = 1}),
    [](const ::testing::TestParamInfo<MatmulFrozenCase>& info) { return info.param.name; });
