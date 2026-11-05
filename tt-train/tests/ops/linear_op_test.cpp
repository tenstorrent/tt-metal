// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/linear_op.hpp"

#include <fmt/ranges.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <tt_stl/reflection.hpp>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/core/xmath.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "init/tensor_initializers.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_sum/device/moreh_sum_device_operation.hpp"

class LinearOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

void compare_tensors(const ttnn::Tensor& t1, const ttnn::Tensor& t2, float eps) {
    ASSERT_EQ(t1.logical_shape(), t2.logical_shape());
    auto t1_vec = ttml::core::to_vector(t1);
    auto t2_vec = ttml::core::to_vector(t2);
    ASSERT_EQ(t1_vec.size(), t2_vec.size());
    auto it = std::mismatch(
        t1_vec.begin(), t1_vec.end(), t2_vec.begin(), [eps](float a, float b) { return std::abs(a - b) <= eps; });
    if (it.first != t1_vec.end()) {
        EXPECT_NEAR(*it.first, *it.second, eps);
    }
    EXPECT_TRUE(it.first == t1_vec.end());
}

bool compare_tensors_for_broken(const ttnn::Tensor& t1, const ttnn::Tensor& t2, float eps) {
    if (t1.logical_shape() != t2.logical_shape()) {
        return false;
    }
    auto t1_vec = ttml::core::to_vector(t1);
    auto t2_vec = ttml::core::to_vector(t2);
    return std::equal(
        t1_vec.begin(), t1_vec.end(), t2_vec.begin(), [eps](float a, float b) { return std::abs(a - b) <= eps; });
}

// Disabled: non-deterministic accuracy failures — https://github.com/tenstorrent/tt-metal/issues/46121
TEST_F(LinearOpTest, DISABLED_TTNNBackwardGoodShape) {
    auto tensor = ttml::autograd::create_tensor();
    ttml::init::uniform_init(tensor, ttnn::Shape({64, 1, 256, 64}), ttml::init::UniformRange{-0.1F, 0.1F});
    tensor->set_requires_grad(true);

    auto weight = ttml::autograd::create_tensor();
    ttml::init::uniform_init(weight, ttnn::Shape({1, 1, 64, 64}), ttml::init::UniformRange{-0.1F, 0.1F});
    weight->set_requires_grad(true);

    auto bias = ttml::autograd::create_tensor();
    ttml::init::uniform_init(bias, ttnn::Shape({1, 1, 1, 64}), ttml::init::UniformRange{-0.1F, 0.1F});
    bias->set_requires_grad(true);

    auto out = ttml::autograd::create_tensor();
    ttml::init::uniform_init(out, ttnn::Shape({64, 1, 256, 64}), ttml::init::UniformRange{-0.1F, 0.1F});
    out->set_grad(out->get_value());
    out->set_requires_grad(true);

    ttml::ops::ttnn_linear_backward(tensor, weight, bias, out);
    auto ttnn_tensor_grad = tensor->get_grad();
    auto ttnn_weight_grad = weight->get_grad();
    auto ttnn_bias_grad = bias->get_grad();
    tensor->set_grad(ttnn::Tensor());
    weight->set_grad(ttnn::Tensor());
    bias->set_grad(ttnn::Tensor());

    ttml::ops::moreh_linear_backward(tensor, weight, bias, out);
    auto moreh_tensor_grad = tensor->get_grad();
    auto moreh_weight_grad = weight->get_grad();
    auto moreh_bias_grad = bias->get_grad();

    const float eps = 3125e-5F;
    compare_tensors(ttnn_tensor_grad, moreh_tensor_grad, eps);
    compare_tensors(ttnn_weight_grad, moreh_weight_grad, eps);
    compare_tensors(ttnn_bias_grad, moreh_bias_grad, eps);
}
void test_linear(uint32_t batch, uint32_t emb_dim) {
    std::cout << "dim: " << emb_dim << std::endl;
    auto tensor = ttml::autograd::create_tensor();
    ttml::init::uniform_init(tensor, ttnn::Shape({batch, 1, 1024, 768}), ttml::init::UniformRange{-0.1F, 0.1F});

    auto weight = ttml::autograd::create_tensor();
    ttml::init::uniform_init(weight, ttnn::Shape({1, 1, emb_dim, 768}), ttml::init::UniformRange{-0.1F, 0.1F});

    auto bias = ttml::autograd::create_tensor();
    ttml::init::uniform_init(bias, ttnn::Shape({1, 1, 1, emb_dim}), ttml::init::UniformRange{-0.1F, 0.1F});

    ttml::ops::linear_op(tensor, weight, bias);
}
TEST_F(LinearOpTest, TTNNLargeLinearOpWithBias) {
    uint32_t dim = 4096;
    uint32_t batch = 32;  // it works with batch = 1, please try to check from 4 to 64
    EXPECT_NO_FATAL_FAILURE(test_linear(batch, 4 * dim));
}

// Currently raises SEGFAULT

// TEST_F(LinearOpTest, TTNNBackwardBadShape_BROKEN) {
//     auto* device = &ttml::autograd::ctx().get_device();
//     auto tensor = ttml::autograd::create_tensor();
//     ttml::init::uniform_init(tensor, ttnn::Shape({128, 1, 1,
//     128}), ttml::init::UniformRange{-0.1F, 0.1F});

//     auto weight = ttml::autograd::create_tensor();
//     ttml::init::uniform_init(weight, ttnn::Shape({1, 1, 256,
//     128}), ttml::init::UniformRange{-0.1F, 0.1F});

//     auto bias = ttml::autograd::create_tensor();
//     ttml::init::uniform_init(bias, ttnn::Shape({1, 1, 1, 256}),
//     ttml::init::UniformRange{-0.1F, 0.1F});

//     auto out = ttml::autograd::create_tensor();
//     ttml::init::uniform_init(out, ttnn::Shape({128, 1, 1, 256}),
//     ttml::init::UniformRange{-0.1F, 0.1F}); out->set_grad(out->get_value());

//     ttml::ops::ttnn_linear_backward(tensor, weight, bias, out);
//     auto ttnn_tensor_grad = tensor->get_grad();
//     auto ttnn_weight_grad = weight->get_grad();
//     auto ttnn_bias_grad = bias->get_grad();
//     tensor->set_grad(ttnn::Tensor());
//     weight->set_grad(ttnn::Tensor());
//     bias->set_grad(ttnn::Tensor());

//     ttml::ops::moreh_linear_backward(tensor, weight, bias, out);
//     auto moreh_tensor_grad = tensor->get_grad();
//     auto moreh_weight_grad = weight->get_grad();
//     auto moreh_bias_grad = bias->get_grad();

//     const float eps = 2e-2F;
//     bool success = compare_tensors_for_broken(ttnn_tensor_grad,
//     moreh_tensor_grad, eps) &&
//                    compare_tensors_for_broken(ttnn_weight_grad,
//                    moreh_weight_grad, eps) &&
//                    compare_tensors_for_broken(ttnn_bias_grad,
//                    moreh_bias_grad, eps);
//     EXPECT_FALSE(success);
// }

namespace {

// out = x @ W^T + b with x: [batch, 1, S, in], W: [1, 1, out, in], b: [1, 1, 1, out]. Each row freezes a different set
// of inputs, so it runs a different subset of ttnn_linear_backward's guarded grad branches.
struct LinearFrozenCase {
    std::string name;
    bool x_requires_grad = true;
    bool w_requires_grad = true;
    bool b_requires_grad = true;
    bool has_bias = true;
};

constexpr uint32_t kBatch = 2U, kSeq = 32U, kIn = 64U, kOut = 96U;

// Small integers keep every product and partial sum exact in bf16: |x|, |W| <= 2 and |out_grad| <= 1 bound dW by
// 2 * batch * S = 128, dx by 2 * out = 192 and db by batch * S = 64, all < 256, so the device grads must equal the host
// reference bit for bit.
xt::xarray<float> make_small_ints(const std::vector<uint32_t>& shape, float bound, uint32_t seed) {
    return xt::round(ttml::test_utils::make_uniform_xarray<float>(shape, -bound, bound, seed));
}

// Copy of t with its leading dims flattened into rows: [..., rows, cols] -> [volume / cols, cols].
xt::xarray<float> flatten_rows(const xt::xarray<float>& t) {
    xt::xarray<float> flat = t;
    flat.reshape(std::vector<size_t>{t.size() / t.shape().back(), t.shape().back()});
    return flat;
}

}  // namespace

class LinearOpFrozenInputTest : public ::testing::TestWithParam<LinearFrozenCase> {
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

// add_grad drops a frozen tensor's grad, so the grads alone cannot show that a frozen input's grad was skipped.
// backward runs once under a NORMAL-mode capture: the ops really execute, giving the grads to check, and the trace
// counts the dW / dx matmuls and the db reduction launched.
TEST_P(LinearOpFrozenInputTest, BackwardComputesOnlyTrainableGrads) {
    const LinearFrozenCase c = GetParam();
    auto* device = &ttml::autograd::ctx().get_device();
    const xt::xarray<float> x = make_small_ints({kBatch, 1U, kSeq, kIn}, /*bound=*/2.4F, /*seed=*/1U);
    const xt::xarray<float> w = make_small_ints({1U, 1U, kOut, kIn}, /*bound=*/2.4F, /*seed=*/2U);
    const xt::xarray<float> b = make_small_ints({1U, 1U, 1U, kOut}, /*bound=*/2.4F, /*seed=*/3U);
    const xt::xarray<float> out_grad = make_small_ints({kBatch, 1U, kSeq, kOut}, /*bound=*/1.4F, /*seed=*/4U);

    const auto t_x = ttml::autograd::create_tensor(ttml::core::from_xtensor(x, device), c.x_requires_grad);
    const auto t_w = ttml::autograd::create_tensor(ttml::core::from_xtensor(w, device), c.w_requires_grad);
    const auto t_b =
        c.has_bias ? ttml::autograd::create_tensor(ttml::core::from_xtensor(b, device), c.b_requires_grad) : nullptr;
    const auto out = ttml::ops::linear_op(t_x, t_w, t_b);

    // With every input frozen add_backward_node records no node, so linear_op's grad function can never run.
    const bool needs_b_grad = c.has_bias && c.b_requires_grad;
    const bool any_trainable = c.x_requires_grad || c.w_requires_grad || needs_b_grad;
    ASSERT_EQ(out->get_node().has_value(), any_trainable) << c.name;
    if (!any_trainable) {
        return;
    }
    out->set_grad(ttml::core::from_xtensor(out_grad, device));

    // The scoped capture ends even if backward throws, so a failing row cannot leave later tests capturing.
    ttnn::graph::ScopedGraphCapture capture(ttnn::graph::GraphProcessor::RunMode::NORMAL);
    out->backward();
    const auto calls = ttnn::graph::extract_calltrace(capture.end_graph_capture());

    // One matmul each for dW and dx, one reduction for db. The names come from the op types, so renaming an op breaks
    // the build rather than silently counting zero.
    const std::string matmul_name{ttsl::get_type_name<ttnn::prim::MatmulDeviceOperation>()};
    const std::string sum_name{ttsl::get_type_name<ttnn::operations::moreh::moreh_sum::MorehSumOperation>()};
    const auto trace_msg = fmt::format("{} backward ops: {}", c.name, fmt::join(calls, ", "));
    EXPECT_EQ(
        std::ranges::count(calls, matmul_name),
        static_cast<int>(c.w_requires_grad) + static_cast<int>(c.x_requires_grad))
        << trace_msg;
    EXPECT_EQ(std::ranges::count(calls, sum_name), static_cast<int>(needs_b_grad)) << trace_msg;

    // With the batch flattened into rows: dW = out_grad^T @ x, dx = out_grad @ W, db = sum of out_grad's rows.
    const xt::xarray<float> x_2d = flatten_rows(x);
    const xt::xarray<float> w_2d = flatten_rows(w);
    const xt::xarray<float> out_grad_2d = flatten_rows(out_grad);
    if (c.w_requires_grad) {
        ASSERT_TRUE(t_w->is_grad_initialized()) << c.name;
        xt::xarray<float> expected = xt::linalg::dot(xt::transpose(out_grad_2d), x_2d);
        expected.reshape(w.shape());
        EXPECT_TRUE(xt::all(xt::equal(ttml::core::to_xtensor(t_w->get_grad()), expected))) << c.name << " grad_w";
    }
    if (c.x_requires_grad) {
        ASSERT_TRUE(t_x->is_grad_initialized()) << c.name;
        xt::xarray<float> expected = xt::linalg::dot(out_grad_2d, w_2d);
        expected.reshape(x.shape());
        EXPECT_TRUE(xt::all(xt::equal(ttml::core::to_xtensor(t_x->get_grad()), expected))) << c.name << " grad_x";
    }
    if (needs_b_grad) {
        ASSERT_TRUE(t_b->is_grad_initialized()) << c.name;
        xt::xarray<float> expected = xt::sum(out_grad_2d, {0});
        expected.reshape(b.shape());
        EXPECT_TRUE(xt::all(xt::equal(ttml::core::to_xtensor(t_b->get_grad()), expected))) << c.name << " grad_b";
    }
}

INSTANTIATE_TEST_SUITE_P(
    LinearFrozenCases,
    LinearOpFrozenInputTest,
    ::testing::Values(
        LinearFrozenCase{.name = "AllTrainable"},
        LinearFrozenCase{
            .name = "AllFrozen", .x_requires_grad = false, .w_requires_grad = false, .b_requires_grad = false},
        LinearFrozenCase{.name = "FrozenX", .x_requires_grad = false},
        LinearFrozenCase{.name = "FrozenW", .w_requires_grad = false},
        LinearFrozenCase{.name = "FrozenB", .b_requires_grad = false},
        LinearFrozenCase{.name = "NoBias", .has_bias = false}),
    [](const ::testing::TestParamInfo<LinearFrozenCase>& info) { return info.param.name; });
