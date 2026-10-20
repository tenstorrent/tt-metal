// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/matmul_op.hpp"

#include <fmt/ranges.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/views/xview.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/graph/graph_operation_queries.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"

namespace {

// out = op(a) @ op(b) with op(a): [M, K], op(b): [K, N]. a is stored as [K, M] when transpose_a, b as [N, K] when
// transpose_b. Each frozen row freezes one input, so it runs exactly one of matmul_op's guarded grad branches;
// the BothTrainable row runs both.
struct MatmulFrozenCase {
    std::string name;
    uint32_t M = 0;
    uint32_t K = 0;
    uint32_t N = 0;
    bool transpose_a = false;
    bool transpose_b = false;
    bool a_requires_grad = true;
    bool b_requires_grad = true;
};

// Small integers keep every product and partial sum exact in bf16: |a|, |b| <= 2 and |out_grad| <= 1 bound each grad
// entry by 2 * max(M, N) = 192 < 256, so the device grads must equal the host reference bit for bit.
xt::xarray<float> make_small_ints(const std::vector<uint32_t>& shape, float bound, uint32_t seed) {
    return xt::round(ttml::test_utils::make_uniform_xarray<float>(shape, -bound, bound, seed));
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
    const std::vector<uint32_t> a_shape = c.transpose_a ? std::vector{c.K, c.M} : std::vector{c.M, c.K};
    const std::vector<uint32_t> b_shape = c.transpose_b ? std::vector{c.N, c.K} : std::vector{c.K, c.N};
    g.a = make_small_ints(a_shape, /*bound=*/2.4F, /*seed=*/1U);
    g.b = make_small_ints(b_shape, /*bound=*/2.4F, /*seed=*/2U);
    g.out_grad = make_small_ints({c.M, c.N}, /*bound=*/1.4F, /*seed=*/3U);
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

// With both inputs frozen, add_backward_node records no node and out does not require grad, so matmul_op's grad
// function, and with it both grad matmuls, can never run: backward has nothing to call and throws.
TEST_F(MatmulOpTest, BothFrozenRecordsNoBackwardNode) {
    const MatmulFrozenCase c{
        .name = "BothFrozen", .M = 32, .K = 64, .N = 96, .a_requires_grad = false, .b_requires_grad = false};
    const MatmulFrozenGraph g = build_graph(c);

    EXPECT_FALSE(g.out->get_requires_grad());
    EXPECT_FALSE(g.out->get_node().has_value()) << "both inputs frozen, but a backward node was recorded";
    EXPECT_THROW(g.out->backward(), std::runtime_error);
    EXPECT_FALSE(g.t_a->is_grad_initialized()) << "frozen a received a gradient";
    EXPECT_FALSE(g.t_b->is_grad_initialized()) << "frozen b received a gradient";
}

namespace {

// out = matmul_op(a, b, transpose_a = true) with a stored as [batch, K, M] and a frozen b = [K, N]. With b frozen,
// backward runs only the dA matmul, which without the batch check returns a scrambled dA and no error.
struct BatchedTransposeAGraph {
    xt::xarray<float> b;
    xt::xarray<float> out_grad;
    ttml::autograd::TensorPtr t_a;
    ttml::autograd::TensorPtr out;
};

BatchedTransposeAGraph build_batched_transpose_a_graph(uint32_t batch, uint32_t M) {
    constexpr uint32_t K = 64U, N = 32U;
    auto* device = &ttml::autograd::ctx().get_device();
    BatchedTransposeAGraph g;
    const xt::xarray<float> a = make_small_ints({batch, K, M}, /*bound=*/2.4F, /*seed=*/1U);
    g.b = make_small_ints({K, N}, /*bound=*/2.4F, /*seed=*/2U);
    g.out_grad = make_small_ints({batch, M, N}, /*bound=*/1.4F, /*seed=*/3U);
    g.t_a = ttml::autograd::create_tensor(ttml::core::from_xtensor(a, device), /*requires_grad=*/true);
    const auto t_b = ttml::autograd::create_tensor(ttml::core::from_xtensor(g.b, device), /*requires_grad=*/false);
    g.out = ttml::ops::matmul_op(g.t_a, t_b, /*transpose_a=*/true, /*transpose_b=*/false);
    g.out->set_grad(ttml::core::from_xtensor(g.out_grad, device));
    return g;
}

}  // namespace

// ttnn's matmul rejects a batched transpose_a forward unless op(a)'s M fits in one tile, so M = 32 is the shape that
// reaches matmul_op's backward.
TEST_F(MatmulOpTest, BatchedTransposeARejectedInBackward) {
    const BatchedTransposeAGraph g = build_batched_transpose_a_graph(/*batch=*/2U, /*M=*/32U);
    EXPECT_THROW(g.out->backward(), std::runtime_error);
    EXPECT_FALSE(g.t_a->is_grad_initialized()) << "a received a gradient from the unsupported backward";
}

// A rank-3 a with a batch of 1 flattens to the 2D case, so the check must let it through with the right dA. a is
// rectangular (stored [64, 96]), so a wrong reshape cannot pass by accident.
TEST_F(MatmulOpTest, BatchOneTransposeAStillSupported) {
    const BatchedTransposeAGraph g = build_batched_transpose_a_graph(/*batch=*/1U, /*M=*/96U);
    g.out->backward();

    // dL/d op(a) = out_grad @ b^T is [M, K]; a is stored as op(a)^T, so dL/da = b @ out_grad^T, [K, M].
    const xt::xarray<float> out_grad_2d = xt::view(g.out_grad, 0, xt::all(), xt::all());
    const xt::xarray<float> expected = xt::linalg::dot(g.b, xt::transpose(out_grad_2d));
    ASSERT_TRUE(g.t_a->is_grad_initialized());
    const xt::xarray<float> grad_a = ttml::core::to_xtensor(g.t_a->get_grad());
    ASSERT_EQ(grad_a.shape(), (std::vector<size_t>{1U, expected.shape(0), expected.shape(1)}));
    EXPECT_TRUE(xt::all(xt::equal(xt::view(grad_a, 0, xt::all(), xt::all()), expected)));
}

TEST_P(MatmulOpFrozenInputTest, FrozenInputGetsNoGradient) {
    const MatmulFrozenCase c = GetParam();
    const MatmulFrozenGraph g = build_graph(c);
    ASSERT_TRUE(g.out->get_requires_grad()) << c.name;
    g.out->backward();

    // dL/d op(a) = out_grad @ op(b)^T and dL/d op(b) = op(a)^T @ out_grad, transposed back to the stored layout.
    const xt::xarray<float> op_a = c.transpose_a ? xt::transpose(g.a) : g.a;
    const xt::xarray<float> op_b = c.transpose_b ? xt::transpose(g.b) : g.b;
    const xt::xarray<float> d_op_a = xt::linalg::dot(g.out_grad, xt::transpose(op_b));
    const xt::xarray<float> d_op_b = xt::linalg::dot(xt::transpose(op_a), g.out_grad);

    if (c.a_requires_grad) {
        ASSERT_TRUE(g.t_a->is_grad_initialized()) << c.name;
        const xt::xarray<float> expected = c.transpose_a ? xt::transpose(d_op_a) : d_op_a;
        EXPECT_TRUE(xt::all(xt::equal(ttml::core::to_xtensor(g.t_a->get_grad()), expected))) << c.name << " grad_a";
    } else {
        EXPECT_FALSE(g.t_a->is_grad_initialized()) << c.name << ": frozen a received a gradient";
    }
    if (c.b_requires_grad) {
        ASSERT_TRUE(g.t_b->is_grad_initialized()) << c.name;
        const xt::xarray<float> expected = c.transpose_b ? xt::transpose(d_op_b) : d_op_b;
        EXPECT_TRUE(xt::all(xt::equal(ttml::core::to_xtensor(g.t_b->get_grad()), expected))) << c.name << " grad_b";
    } else {
        EXPECT_FALSE(g.t_b->is_grad_initialized()) << c.name << ": frozen b received a gradient";
    }
}

// add_grad already drops a frozen tensor's grad, so FrozenInputGetsNoGradient passes even if that grad is computed.
// Capturing backward's device ops (NO_DISPATCH: recorded, not run) shows the frozen side's matmul is never launched.
TEST_P(MatmulOpFrozenInputTest, FrozenInputSkipsItsMatmul) {
    const MatmulFrozenCase c = GetParam();
    const MatmulFrozenGraph g = build_graph(c);

    const auto trace = ttnn::graph::query_trace([&] {
        g.out->backward();
        return 0;
    });
    const auto calls = ttnn::graph::extract_calltrace(trace);
    const auto num_matmuls = std::ranges::count(calls, std::string{"MatmulDeviceOperation"});

    // One grad matmul per trainable input: 2 with both trainable, 1 with one frozen.
    const auto expected_matmuls = static_cast<int>(c.a_requires_grad) + static_cast<int>(c.b_requires_grad);
    EXPECT_EQ(num_matmuls, expected_matmuls)
        << c.name << " backward ops: " << fmt::format("{}", fmt::join(calls, ", "));
}

// transpose_a rows use a rectangular a (stored [K, M] = [64, 96]), so a wrong dL/dout reshape cannot pass by accident.
INSTANTIATE_TEST_SUITE_P(
    MatmulFrozenCases,
    MatmulOpFrozenInputTest,
    ::testing::Values(
        MatmulFrozenCase{.name = "BothTrainable", .M = 32, .K = 64, .N = 96},
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
            .a_requires_grad = false}),
    [](const ::testing::TestParamInfo<MatmulFrozenCase>& info) { return info.param.name; });
