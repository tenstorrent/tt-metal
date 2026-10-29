// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "trivial_ttnn_ops.hpp"

#include <optional>
#include <vector>

#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/gumbel_sample/gumbel_sample.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/moreh/moreh_mean/moreh_mean.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
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

LogSoftmaxContext log_softmax_with_context(const ttnn::Tensor& t, int dim) {
    const auto rank = static_cast<int>(t.logical_shape().rank());
    TT_FATAL(dim >= -rank && dim < rank, "log_softmax: dimension {} out of range for rank {}", dim, rank);
    dim = dim < 0 ? dim + rank : dim;
    TT_FATAL(t.logical_shape()[dim] > 0, "log_softmax: reduction dimension must be nonempty");
    TT_FATAL(
        t.dtype() == ttnn::DataType::BFLOAT16 || t.dtype() == ttnn::DataType::FLOAT32,
        "log_softmax: expected BFLOAT16 or FLOAT32 input");

    auto input = ttnn::typecast(t, ttnn::DataType::FLOAT32);
    auto t_max = ttnn::max(input, dim, /* keepdim */ true, std::nullopt, core::ComputeKernelConfig::precise());
    auto shifted = ttnn::subtract(input, t_max);

    // Select exactly one maximum, including for tied logits. FP32 indices are exact
    // up to 2^24; BF16 indices would alias adjacent classes beyond 256.
    TT_FATAL(t.logical_shape()[dim] <= (1U << 24), "log_softmax: reduction dimension exceeds exact FP32 index range");
    auto indices = ttnn::typecast(
        ttnn::to_layout(ttnn::argmax(input, dim, /* keepdim */ true), ttnn::Layout::TILE), ttnn::DataType::FLOAT32);
    auto positions = ttnn::arange(t.logical_shape()[dim], ttnn::DataType::FLOAT32, *t.device(), t.memory_config());
    std::vector<uint32_t> index_shape(rank, 1);
    index_shape[dim] = t.logical_shape()[dim];
    positions = ttnn::to_layout(ttnn::reshape(positions, ttnn::Shape(index_shape)), ttnn::Layout::TILE);
    auto max_mask = ttnn::eq(indices, positions);

    // Eq. (5), https://arxiv.org/abs/2610.31642: exclude the selected e^0 = 1
    // before summing. log1p preserves the tail even when 1 + tail rounds to 1.
    auto others = ttnn::where(max_mask, 0.F, ttnn::exp(shifted));
    auto log_normalizer = ttnn::log1p(sum_ttnn(others, dim, /* keep_dim */ true));
    return {ttnn::subtract(shifted, log_normalizer), max_mask};
}

ttnn::Tensor log_softmax(const ttnn::Tensor& t, int dim) {
    return ttnn::typecast(log_softmax_with_context(t, dim).log_probabilities, t.dtype());
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

    // The fused op requires the mask to match the logits dtype; ttml::metal::gumbel_sample owns
    // that normalization (it typecasts a mismatched mask), so the mask is passed through as built.
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
