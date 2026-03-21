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
        ttnn::sum(squares, ttsl::SmallVector<int>{-2, -1}, true, std::nullopt, core::ComputeKernelConfig::precise());
    ttnn::Tensor norm_tensor = ttnn::sqrt(sum_squares);

    ttnn::Tensor norm_plus_eps = ttnn::add(norm_tensor, eps);
    X = ttnn::divide(X, norm_plus_eps);

    auto shape = X.logical_shape();
    const uint32_t m = shape[-2];
    const uint32_t n = shape[-1];
    const bool needs_transpose = (m > n);

    if (steps <= 0) {
        return X;
    }

    if (needs_transpose) {
        X = ttnn::transpose(X, -2, -1);
        shape = X.logical_shape();
    }

    // Preallocate buffers: square shape [batch..., m_eff, m_eff] and X shape [batch..., m_eff, n_eff]
    const auto rank = shape.rank();
    std::vector<uint32_t> mm_dims;
    for (uint32_t i = 0; i < rank; ++i) {
        mm_dims.push_back(shape[i]);
    }
    mm_dims.back() = mm_dims[rank - 2];
    ttnn::Shape mm_shape(mm_dims);

    auto* device = X.device();
    auto buf_A = ttnn::empty(mm_shape, X.dtype(), X.layout(), device, X.memory_config());
    auto buf_A2 = ttnn::empty(mm_shape, X.dtype(), X.layout(), device, X.memory_config());
    auto buf_BX = ttnn::empty(shape, X.dtype(), X.layout(), device, X.memory_config());

    for (int iter = 0; iter < steps; ++iter) {
        // A = X @ X^T
        ttnn_fixed::matmul(X, X, false, true, buf_A);

        // A^2 = A @ A
        ttnn_fixed::matmul(buf_A, buf_A, false, false, buf_A2);

        // B = b*A + c*A^2: first scale A by b in-place, then fuse add with alpha
        ttnn::multiply(buf_A, b, std::nullopt, std::nullopt, buf_A);
        ttnn::addalpha(buf_A, buf_A2, c, std::nullopt, buf_A);

        // X_new = a*X + B @ X: compute B @ X, then fuse add with alpha
        ttnn_fixed::matmul(buf_A, X, false, false, buf_BX);
        ttnn::addalpha(buf_BX, X, a, std::nullopt, X);
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
