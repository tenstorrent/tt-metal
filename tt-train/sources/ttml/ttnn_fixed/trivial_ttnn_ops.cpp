// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "trivial_ttnn_ops.hpp"

#include <core/ttnn_all_includes.hpp>
#include <ttnn/operations/moreh/moreh_sum/moreh_sum.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ttnn_fixed {

tt::tt_metal::Tensor sum_over_dim(const tt::tt_metal::Tensor& t, uint32_t dim) {
    return sum_ttnn(t, dim, /* keepdim */ true);
}

tt::tt_metal::Tensor sum_over_batch(const tt::tt_metal::Tensor& t) {
    return sum_over_dim(t, /* dim */ 0);
}

// Stable log-softmax implementation
tt::tt_metal::Tensor log_softmax(const tt::tt_metal::Tensor& t, int dim) {
    auto t_max = ttnn::max(t, dim, /* keepdim */ true);
    auto t_sub_max = ttnn::subtract(t, t_max);

    auto t_sub_max_exp = ttnn::exp(t_sub_max);
    auto t_sum_over_dim = sum_over_dim(t_sub_max_exp, dim);

    auto log_t_sum_over_dim = ttnn::log(t_sum_over_dim);
    return ttnn::subtract(t_sub_max, log_t_sum_over_dim);
}

// Stable softmax implementation
// ttnn::softmax also exists, but it is not stable (even after max subtraction optimization)
tt::tt_metal::Tensor softmax(const tt::tt_metal::Tensor& t, int dim) {
    return ttnn::softmax(
        t,
        /* dim */ dim,
        /*memory_config */ std::nullopt,
        ttml::core::ComputeKernelConfig::softmax(),
        /*stable*/ true);
}

tt::tt_metal::Tensor divide(const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b) {
    auto inv_b = ttnn::reciprocal(/* queue_id */ 0, b);
    return ttnn::multiply(a, inv_b);
}

tt::tt_metal::Tensor mean_moreh(const tt::tt_metal::Tensor& t, int dim, bool keep_dim) {
    auto res = ttnn::moreh_mean(
        t,
        dim,
        keep_dim,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        /* device_compute_kernel_config */ core::ComputeKernelConfig::precise());
    return res;
}
tt::tt_metal::Tensor mean_ttnn(const tt::tt_metal::Tensor& t, int dim, bool keep_dim) {
    return ttnn::mean(t, dim, keep_dim, std::nullopt, core::ComputeKernelConfig::precise());
}

tt::tt_metal::Tensor sum_moreh(const tt::tt_metal::Tensor& t, int dim, bool keep_dim) {
    auto res = ttnn::moreh_sum(
        t,
        dim,
        keep_dim,
        std::nullopt,
        std::nullopt,
        /* device_compute_kernel_config */ core::ComputeKernelConfig::precise());
    return res;
}
tt::tt_metal::Tensor sum_ttnn(const tt::tt_metal::Tensor& t, int dim, bool keep_dim) {
    return ttnn::sum(t, dim, keep_dim, std::nullopt, core::ComputeKernelConfig::precise());
}

tt::tt_metal::Tensor ttnn_matmul(
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    bool transpose_a,
    bool transpose_b,
    const ttnn::WormholeComputeKernelConfig& config) {
    return ttnn::matmul(
        a,
        b,
        transpose_a,
        transpose_b,
        /* memory_config */ std::nullopt,
        /* dtype */ std::nullopt,
        /* program_config */ std::nullopt,
        /* activation */ std::nullopt,
        /* compute_kernel_config */
        config,
        /* core_grid */ ttnn::CoreGrid{7, 8},
        /* output_tile */ std::nullopt);
}

tt::tt_metal::Tensor moreh_matmul(
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    bool transpose_a,
    bool transpose_b,
    const ttnn::WormholeComputeKernelConfig& config) {
    return ttnn::moreh_matmul(
        a,
        b,
        transpose_a,
        transpose_b,
        /* memory_config */ std::nullopt,
        std::nullopt,
        std::nullopt,
        config);
}

tt::tt_metal::Tensor matmul(
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    bool transpose_a,
    bool transpose_b,
    const ttnn::WormholeComputeKernelConfig& config,
    bool use_moreh) {
    if (use_moreh) {
        return moreh_matmul(a, b, transpose_a, transpose_b, config);
    } else {
        return ttnn_matmul(a, b, transpose_a, transpose_b, config);
    }
}

}  // namespace ttml::ttnn_fixed
