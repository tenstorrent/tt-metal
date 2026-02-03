// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "newton_schulz_op.hpp"

#include <core/ttnn_all_includes.hpp>

#include "core/compute_kernel_config.hpp"
#include "ttnn_fixed/matmuls.hpp"

namespace ttml::ops {

tt::tt_metal::Tensor newtonschulz(const tt::tt_metal::Tensor& G, int steps, float eps, float a, float b, float c) {
    ttnn::Tensor X = G;

    ttnn::Tensor squares = ttnn::square(X);
    ttnn::Tensor sum_squares =
        ttnn::sum(squares, ttnn::SmallVector<int>{-2, -1}, true, std::nullopt, core::ComputeKernelConfig::precise());
    ttnn::Tensor norm_tensor = ttnn::sqrt(sum_squares);

    ttnn::Tensor norm_plus_eps = ttnn::add(norm_tensor, eps);
    X = ttnn::divide(X, norm_plus_eps);

    auto shape = X.logical_shape();
    uint32_t m = shape[-2];
    uint32_t n = shape[-1];
    bool needs_transpose = (m > n);

    if (needs_transpose) {
        X = ttnn::transpose(X, -2, -1);
    }

    for (int iter = 0; iter < steps; ++iter) {
        ttnn::Tensor A = ttnn_fixed::matmul(X, X, false, true);

        ttnn::Tensor b_A = ttnn::multiply(A, b);
        ttnn::Tensor A_squared = ttnn_fixed::matmul(A, A, false, false);
        ttnn::Tensor c_A_squared = ttnn::multiply(A_squared, c);
        ttnn::Tensor B = ttnn::add(b_A, c_A_squared);

        ttnn::Tensor a_X = ttnn::multiply(X, a);
        ttnn::Tensor B_X = ttnn_fixed::matmul(B, X, false, false);
        X = ttnn::add(a_X, B_X);
    }

    if (needs_transpose) {
        X = ttnn::transpose(X, -2, -1);
    }

    return X;
}

tt::tt_metal::Tensor newtonschulz5(const tt::tt_metal::Tensor& G, int steps, float eps) {
    return newtonschulz(G, steps, eps, 3.4445f, -4.7750f, 2.0315f);
}

}  // namespace ttml::ops
