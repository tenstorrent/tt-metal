// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core/ttnn_all_includes.hpp>

#include "core/tt_tensor_utils.hpp"

namespace ttml::ttnn_fixed {

tt::tt_metal::Tensor sum_over_dim(const tt::tt_metal::Tensor& t, uint32_t dim);
tt::tt_metal::Tensor sum_over_batch(const tt::tt_metal::Tensor& t);
tt::tt_metal::Tensor log_softmax(const tt::tt_metal::Tensor& t, int dim);
tt::tt_metal::Tensor softmax(const tt::tt_metal::Tensor& t, int dim);
tt::tt_metal::Tensor divide(const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b);

tt::tt_metal::Tensor mean_moreh(const tt::tt_metal::Tensor& t, int dim, bool keep_dim);
tt::tt_metal::Tensor mean_ttnn(const tt::tt_metal::Tensor& t, int dim, bool keep_dim);

tt::tt_metal::Tensor sum_moreh(const tt::tt_metal::Tensor& t, int dim, bool keep_dim);
tt::tt_metal::Tensor sum_ttnn(const tt::tt_metal::Tensor& t, int dim, bool keep_dim);

tt::tt_metal::Tensor ttnn_matmul(
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    bool transpose_a,
    bool transpose_b,
    const ttnn::WormholeComputeKernelConfig& config);

tt::tt_metal::Tensor moreh_matmul(
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    bool transpose_a,
    bool transpose_b,
    const ttnn::WormholeComputeKernelConfig& config);

tt::tt_metal::Tensor matmul(
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    bool transpose_a,
    bool transpose_b,
    const ttnn::WormholeComputeKernelConfig& config,
    bool use_moreh = false);

}  // namespace ttml::ttnn_fixed
