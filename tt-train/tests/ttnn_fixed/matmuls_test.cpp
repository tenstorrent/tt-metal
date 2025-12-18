// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn_fixed/matmuls.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"

class MatmulsTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

namespace {

[[maybe_unused]] xt::xarray<float> matmul(
    const xt::xarray<float>& a, const xt::xarray<float>& b, bool transpose_a, bool transpose_b) {
    // Conditionally transpose the inputs
    auto A = transpose_a ? xt::transpose(a) : a;
    auto B = transpose_b ? xt::transpose(b) : b;
    // Perform matrix multiplication using xtensor-blas dot
    return xt::linalg::dot(A, B);
}

// Backward pass for matrix multiplication.
// Given out_grad = dL/dy, compute gradients with respect to a and b.
[[maybe_unused]] std::pair<xt::xarray<float>, xt::xarray<float>> matmul_backward(
    const xt::xarray<float>& a,
    const xt::xarray<float>& b,
    const xt::xarray<float>& out_grad,
    bool transpose_a,
    bool transpose_b) {
    xt::xarray<float> grad_a, grad_b;

    if (!transpose_a && !transpose_b) {
        // y = a * b
        // dL/da = out_grad * b^T, dL/db = a^T * out_grad
        grad_a = xt::linalg::dot(out_grad, xt::transpose(b));
        grad_b = xt::linalg::dot(xt::transpose(a), out_grad);
    } else if (!transpose_a && transpose_b) {
        // y = a * b^T
        // dL/da = out_grad * b, dL/db = (out_grad)^T * a
        grad_a = xt::linalg::dot(out_grad, b);
        grad_b = xt::linalg::dot(xt::transpose(out_grad), a);
    } else if (transpose_a && !transpose_b) {
        // y = a^T * b
        // dL/da = b * (out_grad)^T, dL/db = a * out_grad
        grad_a = xt::linalg::dot(b, xt::transpose(out_grad));
        grad_b = xt::linalg::dot(a, out_grad);
    } else {  // transpose_a && transpose_b
        // y = a^T * b^T
        // dL/da = (b^T) * (out_grad)^T, dL/db = (out_grad)^T * (a^T)
        grad_a = xt::linalg::dot(xt::transpose(b), xt::transpose(out_grad));
        grad_b = xt::linalg::dot(xt::transpose(out_grad), xt::transpose(a));
    }

    return std::make_pair(grad_a, grad_b);
}

}  // namespace

namespace ttml::ttnn_fixed::tests {

TEST_F(MatmulsTest, MatMulNoTranspose) {
    // Create simple matrices.
    xt::xarray<float> a = {{1, 2}, {3, 4}};
    xt::xarray<float> b = {{5, 6}, {7, 8}};

    // Convert to ttnn tensors.
    auto t_a = core::from_xtensor(a, &autograd::ctx().get_device());
    auto t_b = core::from_xtensor(b, &autograd::ctx().get_device());

    // Compute the result using the ttnn op.
    auto t_y = matmul(t_a, t_b, false, false);
    xt::xarray<float> y = core::to_xtensor(t_y);

    // Compute the expected result using xtensor goldens.
    xt::xarray<float> expected = xt::linalg::dot(a, b);
    EXPECT_TRUE(xt::allclose(y, expected));
}

TEST_F(MatmulsTest, MatMulTransposeA) {
    xt::xarray<float> a = {{1, 2}, {3, 4}};
    xt::xarray<float> b = {{5, 6}, {7, 8}};

    auto t_a = core::from_xtensor(a, &autograd::ctx().get_device());
    auto t_b = core::from_xtensor(b, &autograd::ctx().get_device());

    // Use transpose_a = true.
    auto t_y = matmul(t_a, t_b, true, false);
    xt::xarray<float> y = core::to_xtensor(t_y);

    // Expected: (a^T) * b.
    xt::xarray<float> expected = xt::linalg::dot(xt::transpose(a), b);
    EXPECT_TRUE(xt::allclose(y, expected));
}

TEST_F(MatmulsTest, MatMulTransposeB) {
    xt::xarray<float> a = {{1, 2}, {3, 4}};
    xt::xarray<float> b = {{5, 6}, {7, 8}};

    auto t_a = core::from_xtensor(a, &autograd::ctx().get_device());
    auto t_b = core::from_xtensor(b, &autograd::ctx().get_device());

    // Use transpose_b = true.
    auto t_y = matmul(t_a, t_b, false, true);
    xt::xarray<float> y = core::to_xtensor(t_y);

    // Expected: a * (b^T).
    xt::xarray<float> expected = xt::linalg::dot(a, xt::transpose(b));
    EXPECT_TRUE(xt::allclose(y, expected));
}

TEST_F(MatmulsTest, MatMulTransposeBoth) {
    xt::xarray<float> a = {{1, 2}, {3, 4}};
    xt::xarray<float> b = {{5, 6}, {7, 8}};

    auto t_a = core::from_xtensor(a, &autograd::ctx().get_device());
    auto t_b = core::from_xtensor(b, &autograd::ctx().get_device());

    // Use both transpositions.
    auto t_y = matmul(t_a, t_b, true, true);
    xt::xarray<float> y = core::to_xtensor(t_y);

    // Expected: (a^T) * (b^T).
    xt::xarray<float> expected = xt::linalg::dot(xt::transpose(a), xt::transpose(b));
    EXPECT_TRUE(xt::allclose(y, expected));
}

TEST_F(MatmulsTest, MatMulBackwardNoTranspose) {
    // Create matrices and an output gradient.
    xt::xarray<float> a = {{1, 2}, {3, 4}};
    xt::xarray<float> b = {{5, 6}, {7, 8}};
    xt::xarray<float> out_grad = {{1, 1}, {1, 1}};

    // Convert to ttnn tensors.
    auto t_a = core::from_xtensor(a, &autograd::ctx().get_device());
    auto t_b = core::from_xtensor(b, &autograd::ctx().get_device());
    auto t_out_grad = core::from_xtensor(out_grad, &autograd::ctx().get_device());

    // Compute the backward pass.
    auto grads = matmul_backward(t_a, t_b, t_out_grad, false, false);
    xt::xarray<float> grad_a = core::to_xtensor(grads.first);
    xt::xarray<float> grad_b = core::to_xtensor(grads.second);

    // Expected gradients:
    // grad_a = out_grad * (b^T) and grad_b = (a^T) * out_grad.
    xt::xarray<float> expected_grad_a = xt::linalg::dot(out_grad, xt::transpose(b));
    xt::xarray<float> expected_grad_b = xt::linalg::dot(xt::transpose(a), out_grad);

    EXPECT_TRUE(xt::allclose(grad_a, expected_grad_a));
    EXPECT_TRUE(xt::allclose(grad_b, expected_grad_b));
}

TEST_F(MatmulsTest, MatMulBackwardTransposeA) {
    // Test backward with transpose_a = true.
    xt::xarray<float> a = {{1, 2}, {3, 4}};
    xt::xarray<float> b = {{5, 6}, {7, 8}};
    xt::xarray<float> out_grad = {{1, 1}, {1, 1}};

    auto t_a = core::from_xtensor(a, &autograd::ctx().get_device());
    auto t_b = core::from_xtensor(b, &autograd::ctx().get_device());
    auto t_out_grad = core::from_xtensor(out_grad, &autograd::ctx().get_device());

    auto grads = matmul_backward(t_a, t_b, t_out_grad, true, false);
    xt::xarray<float> grad_a = core::to_xtensor(grads.first);
    xt::xarray<float> grad_b = core::to_xtensor(grads.second);

    // For y = (a^T) * b:
    // Expected gradients:
    // grad_a = b * (out_grad)^T, and grad_b = a * out_grad.
    xt::xarray<float> expected_grad_a = xt::linalg::dot(b, xt::transpose(out_grad));
    xt::xarray<float> expected_grad_b = xt::linalg::dot(a, out_grad);

    EXPECT_TRUE(xt::allclose(grad_a, expected_grad_a));
    EXPECT_TRUE(xt::allclose(grad_b, expected_grad_b));
}

TEST_F(MatmulsTest, MatMulBackwardTransposeB) {
    // Test backward with transpose_b = true.
    xt::xarray<float> a = {{1, 2}, {3, 4}};
    xt::xarray<float> b = {{5, 6}, {7, 8}};
    xt::xarray<float> out_grad = {{1, 1}, {1, 1}};

    auto t_a = core::from_xtensor(a, &autograd::ctx().get_device());
    auto t_b = core::from_xtensor(b, &autograd::ctx().get_device());
    auto t_out_grad = core::from_xtensor(out_grad, &autograd::ctx().get_device());

    auto grads = matmul_backward(t_a, t_b, t_out_grad, false, true);
    xt::xarray<float> grad_a = core::to_xtensor(grads.first);
    xt::xarray<float> grad_b = core::to_xtensor(grads.second);

    // For y = a * (b^T):
    // Expected gradients:
    // grad_a = out_grad * b, and grad_b = (out_grad)^T * a.
    xt::xarray<float> expected_grad_a = xt::linalg::dot(out_grad, b);
    xt::xarray<float> expected_grad_b = xt::linalg::dot(xt::transpose(out_grad), a);

    EXPECT_TRUE(xt::allclose(grad_a, expected_grad_a));
    EXPECT_TRUE(xt::allclose(grad_b, expected_grad_b));
}

TEST_F(MatmulsTest, MatMulBackwardTransposeBoth) {
    // Test backward with both transpose flags true.
    xt::xarray<float> a = {{1, 2}, {3, 4}};
    xt::xarray<float> b = {{5, 6}, {7, 8}};
    xt::xarray<float> out_grad = {{1, 1}, {1, 1}};

    auto t_a = core::from_xtensor(a, &autograd::ctx().get_device());
    auto t_b = core::from_xtensor(b, &autograd::ctx().get_device());
    auto t_out_grad = core::from_xtensor(out_grad, &autograd::ctx().get_device());

    auto grads = matmul_backward(t_a, t_b, t_out_grad, true, true);
    xt::xarray<float> grad_a = core::to_xtensor(grads.first);
    xt::xarray<float> grad_b = core::to_xtensor(grads.second);

    // For y = (a^T) * (b^T):
    // Expected gradients:
    // grad_a = (b^T) * (out_grad)^T, and grad_b = (out_grad)^T * (a^T).
    xt::xarray<float> expected_grad_a = xt::linalg::dot(xt::transpose(b), xt::transpose(out_grad));
    xt::xarray<float> expected_grad_b = xt::linalg::dot(xt::transpose(out_grad), xt::transpose(a));

    EXPECT_TRUE(xt::allclose(grad_a, expected_grad_a));
    EXPECT_TRUE(xt::allclose(grad_b, expected_grad_b));
}

struct MatMulManualVsFlagParams {
    std::vector<size_t> shape_a;
    std::vector<size_t> shape_b;
    bool transpose_a;
    bool transpose_b;

    friend std::ostream& operator<<(std::ostream& os, const MatMulManualVsFlagParams& p) {
        os << "a=[";
        for (size_t i = 0; i < p.shape_a.size(); ++i) {
            os << p.shape_a[i] << (i + 1 < p.shape_a.size() ? "," : "");
        }
        os << "]_b=[";
        for (size_t i = 0; i < p.shape_b.size(); ++i) {
            os << p.shape_b[i] << (i + 1 < p.shape_b.size() ? "," : "");
        }
        os << "]_transA=" << p.transpose_a << "_transB=" << p.transpose_b;
        return os;
    }
};

class MatMulManualVsFlagTest : public ::testing::TestWithParam<MatMulManualVsFlagParams> {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_P(MatMulManualVsFlagTest, ManualTransposeMatchesFlag) {
    const auto& params = GetParam();

    xt::xarray<float>::shape_type shape_a(params.shape_a.begin(), params.shape_a.end());
    xt::xarray<float>::shape_type shape_b(params.shape_b.begin(), params.shape_b.end());
    xt::xarray<float> a = xt::random::rand<float>(shape_a, -1.0F, 1.0F);
    xt::xarray<float> b = xt::random::rand<float>(shape_b, -1.0F, 1.0F);

    auto t_a = ttml::core::from_xtensor(a, &ttml::autograd::ctx().get_device());
    auto t_b = ttml::core::from_xtensor(b, &ttml::autograd::ctx().get_device());

    // Manual transpose approach
    auto t_a_for_manual = params.transpose_a ? ttnn::transpose(t_a, -2, -1) : t_a;
    auto t_b_for_manual = params.transpose_b ? ttnn::transpose(t_b, -2, -1) : t_b;
    auto t_y_manual = ttml::ttnn_fixed::matmul(t_a_for_manual, t_b_for_manual, false, false);
    xt::xarray<float> y_manual = ttml::core::to_xtensor(t_y_manual);

    // Flag-based approach
    auto t_y_flag = ttml::ttnn_fixed::matmul(t_a, t_b, params.transpose_a, params.transpose_b);
    xt::xarray<float> y_flag = ttml::core::to_xtensor(t_y_flag);

    std::cout << "y_manual shape: " << xt::adapt(y_manual.shape()) << std::endl;
    std::cout << "y_flag shape: " << xt::adapt(y_flag.shape()) << std::endl;

    auto num_diff = xt::sum(xt::not_equal(y_manual, y_flag))();
    std::cout << "Number of differing elements: " << num_diff << " / " << y_manual.size() << std::endl;

    EXPECT_EQ(y_manual.shape(), y_flag.shape());
    EXPECT_EQ(y_manual, y_flag);
}

INSTANTIATE_TEST_SUITE_P(
    MatMulTransposeTests,
    MatMulManualVsFlagTest,
    ::testing::Values(
        // a: shape=[64, 6, 256, 64], transpose_a=false | b: shape=[64, 6, 256, 64], transpose_b=true
        MatMulManualVsFlagParams{{64, 6, 256, 64}, {64, 6, 256, 64}, false, true},
        // a: shape=[64, 6, 256, 256], transpose_a=true | b: shape=[64, 6, 256, 64], transpose_b=false
        MatMulManualVsFlagParams{{64, 6, 256, 256}, {64, 6, 256, 64}, true, false},
        // a: shape=[16384, 96], transpose_a=true | b: shape=[16384, 384], transpose_b=false
        MatMulManualVsFlagParams{{16384, 96}, {16384, 384}, true, false},
        // a: shape=[16384, 384], transpose_a=true | b: shape=[16384, 1536], transpose_b=false
        MatMulManualVsFlagParams{{16384, 384}, {16384, 1536}, true, false},
        // a: shape=[16384, 1536], transpose_a=true | b: shape=[16384, 384], transpose_b=false
        MatMulManualVsFlagParams{{16384, 1536}, {16384, 384}, true, false},
        // a: shape=[16384, 384], transpose_a=true | b: shape=[16384, 384], transpose_b=false
        MatMulManualVsFlagParams{{16384, 384}, {16384, 384}, true, false},
        // a: shape=[16384, 1152], transpose_a=true | b: shape=[16384, 384], transpose_b=false
        MatMulManualVsFlagParams{{16384, 1152}, {16384, 384}, true, false}));

}  // namespace ttml::ttnn_fixed::tests
