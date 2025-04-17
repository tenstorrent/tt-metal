// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "trivial_ttnn_ops.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ttnn_fixed {

tt::tt_metal::Tensor sum_over_dim(const tt::tt_metal::Tensor& t, uint32_t dim) {
    return sum_moreh(t, dim, /* keepdim */ true);
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
    auto inv_b = ttnn::reciprocal(ttnn::DefaultQueueId, b);
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
    return ttnn::moreh_sum(
        t,
        dim,
        keep_dim,
        std::nullopt,
        std::nullopt,
        /* device_compute_kernel_config */ core::ComputeKernelConfig::precise());
}
tt::tt_metal::Tensor sum_ttnn(const tt::tt_metal::Tensor& t, int dim, bool keep_dim) {
    return ttnn::sum(t, dim, keep_dim, std::nullopt, core::ComputeKernelConfig::precise());
}

tt::tt_metal::Tensor to_l1_interleaved(const tt::tt_metal::Tensor& t) {
    return ttnn::to_memory_config(t, ttnn::L1_MEMORY_CONFIG);
}

tt::tt_metal::Tensor to_dram_interleaved(const tt::tt_metal::Tensor& t) {
    return ttnn::to_memory_config(t, ttnn::DRAM_MEMORY_CONFIG);
}

}  // namespace ttml::ttnn_fixed
