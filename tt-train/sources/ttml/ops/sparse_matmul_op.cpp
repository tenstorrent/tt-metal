// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_matmul_op.hpp"

#include <stdexcept>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn_fixed/matmuls.hpp"

namespace ttml::ops {

namespace {

struct ShapeInfo {
    uint32_t T_cap;
    uint32_t K;
    uint32_t E;
    uint32_t N;
};

ShapeInfo extract_shapes(const ttnn::Tensor& X, const ttnn::Tensor& W) {
    const auto x_shape = X.logical_shape();
    const auto w_shape = W.logical_shape();
    if (w_shape[-2] != x_shape[-1]) {
        throw std::runtime_error("sparse_matmul: W inner dim must equal X's K.");
    }
    return {x_shape[-2], x_shape[-1], w_shape[0], w_shape[-1]};
}

std::vector<uint32_t> read_offsets_host(const ttnn::Tensor& offsets, uint32_t E, uint32_t T_cap) {
    auto host = offsets.to_vector<uint32_t>();
    if (host.size() != E + 1U) {
        throw std::runtime_error("sparse_matmul: offsets size must be E_local + 1.");
    }
    if (host.back() != T_cap) {
        throw std::runtime_error("sparse_matmul: offsets[-1] must equal T_cap.");
    }
    return host;
}

}  // namespace

ttnn::Tensor sparse_matmul_forward(const ttnn::Tensor& X, const ttnn::Tensor& W, const ttnn::Tensor& offsets) {
    const auto s = extract_shapes(X, W);
    const auto offsets_host = read_offsets_host(offsets, s.E, s.T_cap);

    std::vector<ttnn::Tensor> parts;
    parts.reserve(s.E);

    const ttsl::SmallVector<uint32_t> w_step = {1U, 1U, 1U};
    const ttsl::SmallVector<uint32_t> x_step = {1U, 1U, 1U, 1U};

    for (uint32_t e = 0; e < s.E; ++e) {
        const uint32_t row_lo = offsets_host[e];
        const uint32_t row_hi = offsets_host[e + 1U];
        if (row_hi < row_lo) {
            throw std::runtime_error("sparse_matmul: offsets are not monotonic.");
        }
        if (row_hi == row_lo) {
            continue;
        }

        const ttsl::SmallVector<uint32_t> x_start = {0U, 0U, row_lo, 0U};
        const ttsl::SmallVector<uint32_t> x_end = {1U, 1U, row_hi, s.K};
        auto X_e = ttnn::slice(X, x_start, x_end, x_step);

        const ttsl::SmallVector<uint32_t> w_start = {e, 0U, 0U};
        const ttsl::SmallVector<uint32_t> w_end = {e + 1U, s.K, s.N};
        auto W_e_3d = ttnn::slice(W, w_start, w_end, w_step);
        auto W_e = W_e_3d.reshape(ttnn::Shape({1U, 1U, s.K, s.N}));

        parts.push_back(ttnn_fixed::matmul(X_e, W_e, /*transpose_a=*/false, /*transpose_b=*/false));
    }

    if (parts.empty()) {
        throw std::runtime_error(
            "sparse_matmul: all experts empty (T_cap == 0); caller must avoid this degenerate case.");
    }
    if (parts.size() == 1U) {
        return parts.front();
    }
    return ttnn::concat(parts, /*dim=*/2);
}

std::pair<ttnn::Tensor, ttnn::Tensor> sparse_matmul_backward(
    const ttnn::Tensor& X, const ttnn::Tensor& W, const ttnn::Tensor& dY, const ttnn::Tensor& offsets) {
    const auto s = extract_shapes(X, W);
    const auto offsets_host = read_offsets_host(offsets, s.E, s.T_cap);

    auto* device = X.device();
    const auto dram = ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM};

    std::vector<ttnn::Tensor> dX_parts;
    std::vector<ttnn::Tensor> dW_parts;
    dX_parts.reserve(s.E);
    dW_parts.reserve(s.E);

    const ttsl::SmallVector<uint32_t> w_step = {1U, 1U, 1U};
    const ttsl::SmallVector<uint32_t> step = {1U, 1U, 1U, 1U};

    for (uint32_t e = 0; e < s.E; ++e) {
        const uint32_t row_lo = offsets_host[e];
        const uint32_t row_hi = offsets_host[e + 1U];

        if (row_hi == row_lo) {
            // Empty expert: zero contribution to dW for this slot, no rows for dX.
            dW_parts.push_back(
                ttnn::zeros(ttnn::Shape({1U, s.K, s.N}), W.dtype(), ttnn::Layout::TILE, std::ref(*device), dram));
            continue;
        }

        const ttsl::SmallVector<uint32_t> x_start = {0U, 0U, row_lo, 0U};
        const ttsl::SmallVector<uint32_t> x_end = {1U, 1U, row_hi, s.K};
        auto X_e = ttnn::slice(X, x_start, x_end, step);

        const ttsl::SmallVector<uint32_t> dy_end = {1U, 1U, row_hi, s.N};
        auto dY_e = ttnn::slice(dY, x_start, dy_end, step);

        const ttsl::SmallVector<uint32_t> w_start = {e, 0U, 0U};
        const ttsl::SmallVector<uint32_t> w_end = {e + 1U, s.K, s.N};
        auto W_e_3d = ttnn::slice(W, w_start, w_end, w_step);
        auto W_e = W_e_3d.reshape(ttnn::Shape({1U, 1U, s.K, s.N}));

        // dX_e = dY_e @ W_e^T  →  [1,1,len,K]
        dX_parts.push_back(ttnn_fixed::matmul(dY_e, W_e, /*transpose_a=*/false, /*transpose_b=*/true));

        // dW_e = X_e^T @ dY_e  →  [1,1,K,N], reshaped to [1,K,N] for concat into [E,K,N]
        auto dW_e_4d = ttnn_fixed::matmul(X_e, dY_e, /*transpose_a=*/true, /*transpose_b=*/false);
        dW_parts.push_back(dW_e_4d.reshape(ttnn::Shape({1U, s.K, s.N})));
    }

    if (dX_parts.empty()) {
        throw std::runtime_error("sparse_matmul_backward: all experts empty (T_cap == 0).");
    }

    auto dX = (dX_parts.size() == 1U) ? dX_parts.front() : ttnn::concat(dX_parts, /*dim=*/2);
    auto dW = (dW_parts.size() == 1U) ? dW_parts.front() : ttnn::concat(dW_parts, /*dim=*/0);
    return {dX, dW};
}

autograd::TensorPtr sparse_matmul(
    const autograd::TensorPtr& X, const autograd::TensorPtr& W, const ttnn::Tensor& offsets) {
    auto y = sparse_matmul_forward(X->get_value(), W->get_value(), offsets);
    auto out = autograd::create_tensor(y);

    autograd::GradFunction grad = [X, W, offsets, out]() mutable {
        auto dY = out->get_grad();
        auto [dX, dW] = sparse_matmul_backward(X->get_value(), W->get_value(), dY, offsets);
        X->add_grad(dX);
        W->add_grad(dW);
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, X, W));
    return out;
}

}  // namespace ttml::ops
