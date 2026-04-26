// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_matmul_op.hpp"

#include <stdexcept>
#include <vector>

#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn_fixed/matmuls.hpp"

namespace ttml::ops {

ttnn::Tensor sparse_matmul(const ttnn::Tensor& X, const ttnn::Tensor& W, const ttnn::Tensor& offsets) {
    const auto x_shape = X.logical_shape();
    const auto w_shape = W.logical_shape();

    const uint32_t T_cap = x_shape[-2];
    const uint32_t K = x_shape[-1];
    const uint32_t E = w_shape[0];
    const uint32_t N = w_shape[-1];

    if (w_shape[-2] != K) {
        throw std::runtime_error("sparse_matmul: W inner dim must equal X's K.");
    }

    const auto offsets_host = offsets.to_vector<uint32_t>();
    if (offsets_host.size() != E + 1U) {
        throw std::runtime_error("sparse_matmul: offsets size must be E_local + 1.");
    }
    if (offsets_host.back() != T_cap) {
        throw std::runtime_error("sparse_matmul: offsets[-1] must equal T_cap.");
    }

    std::vector<ttnn::Tensor> parts;
    parts.reserve(E);

    const ttsl::SmallVector<uint32_t> w_step = {1U, 1U, 1U};
    const ttsl::SmallVector<uint32_t> x_step = {1U, 1U, 1U, 1U};

    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t row_lo = offsets_host[e];
        const uint32_t row_hi = offsets_host[e + 1U];
        if (row_hi < row_lo) {
            throw std::runtime_error("sparse_matmul: offsets are not monotonic.");
        }
        if (row_hi == row_lo) {
            continue;
        }

        const ttsl::SmallVector<uint32_t> x_start = {0U, 0U, row_lo, 0U};
        const ttsl::SmallVector<uint32_t> x_end = {1U, 1U, row_hi, K};
        auto X_e = ttnn::slice(X, x_start, x_end, x_step);

        const ttsl::SmallVector<uint32_t> w_start = {e, 0U, 0U};
        const ttsl::SmallVector<uint32_t> w_end = {e + 1U, K, N};
        auto W_e_3d = ttnn::slice(W, w_start, w_end, w_step);
        auto W_e = W_e_3d.reshape(ttnn::Shape({1U, 1U, K, N}));

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

}  // namespace ttml::ops
