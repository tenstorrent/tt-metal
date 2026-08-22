// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "trivial_ttnn_ops.hpp"

#include <optional>
#include <vector>

#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/gumbel_sample/gumbel_sample.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/moreh/moreh_mean/moreh_mean.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttml::ttnn_fixed {

ttnn::Tensor sum_over_dim(const ttnn::Tensor& t, uint32_t dim) {
    return sum_moreh(t, dim, /* keepdim */ true);
}

ttnn::Tensor sum_over_batch(const ttnn::Tensor& t) {
    return sum_over_dim(t, /* dim */ 0);
}

// Stable log-softmax implementation
ttnn::Tensor log_softmax(const ttnn::Tensor& t, int dim) {
    auto t_max = ttnn::max(t, dim, /* keepdim */ true);
    auto t_sub_max = ttnn::subtract(t, t_max);

    auto t_sub_max_exp = ttnn::exp(t_sub_max);
    auto t_sum_over_dim = sum_over_dim(t_sub_max_exp, dim);

    auto log_t_sum_over_dim = ttnn::log(t_sum_over_dim, /*fast_and_approximate_mode=*/true);
    return ttnn::subtract(t_sub_max, log_t_sum_over_dim);
}

// Stable softmax implementation
// ttnn::softmax also exists, but it is not stable (even after max subtraction optimization)
ttnn::Tensor softmax(const ttnn::Tensor& t, int dim) {
    return ttnn::softmax(
        t,
        /* dim */ dim,
        /*memory_config */ std::nullopt,
        ttml::core::ComputeKernelConfig::softmax(),
        /*stable*/ true);
}

ttnn::Tensor divide(const ttnn::Tensor& a, const ttnn::Tensor& b) {
    auto inv_b = ttnn::reciprocal(b);
    return ttnn::multiply(a, inv_b);
}

ttnn::Tensor mean_moreh(const ttnn::Tensor& t, int dim, bool keep_dim) {
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
ttnn::Tensor mean_ttnn(const ttnn::Tensor& t, int dim, bool keep_dim) {
    return ttnn::mean(t, dim, keep_dim, std::nullopt, core::ComputeKernelConfig::precise());
}

ttnn::Tensor sum_moreh(const ttnn::Tensor& t, int dim, bool keep_dim) {
    return ttnn::moreh_sum(
        t,
        dim,
        keep_dim,
        std::nullopt,
        std::nullopt,
        /* device_compute_kernel_config */ core::ComputeKernelConfig::precise());
}
ttnn::Tensor sum_ttnn(const ttnn::Tensor& t, int dim, bool keep_dim) {
    return ttnn::sum(t, dim, keep_dim, std::nullopt, core::ComputeKernelConfig::precise());
}

ttnn::Tensor sample(
    const ttnn::Tensor& t,
    float temperature,
    uint32_t seed,
    std::optional<ttnn::Tensor> logits_mask,
    std::optional<std::vector<uint32_t>> seed_axes,
    std::optional<ttnn::Tensor> positions) {
    // `seed_axes` lists the mesh axes whose devices hold DISTINCT data and must therefore draw
    // DISTINCT noise -- the data-parallel axes (dp / fsdp). Axes left out are treated as replicated
    // (tp) and draw IDENTICAL noise, which is what keeps a replica group agreeing on the token it
    // shares. std::nullopt (the default) seeds no axis, so every device draws the same noise.
    // Callers that need per-device sampling (e.g. GRPO, to avoid duplicate completions across data-
    // parallel ranks) MUST pass their sharded axes explicitly.
    return ttml::metal::gumbel_sample(
        t, temperature, seed, seed_axes.value_or(std::vector<uint32_t>{}), logits_mask, positions);
}

ttnn::Tensor to_l1_interleaved(const ttnn::Tensor& t) {
    return ttnn::to_memory_config(t, ttnn::L1_MEMORY_CONFIG);
}

ttnn::Tensor to_dram_interleaved(const ttnn::Tensor& t) {
    return ttnn::to_memory_config(t, ttnn::DRAM_MEMORY_CONFIG);
}

}  // namespace ttml::ttnn_fixed
