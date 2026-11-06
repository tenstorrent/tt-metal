// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "trivial_ttnn_ops.hpp"

#include <optional>
#include <tt-metalium/constants.hpp>
#include <vector>

#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/gumbel_sample/gumbel_sample.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
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

ttnn::Tensor sum_over_leading_dims(const ttnn::Tensor& t) {
    const auto& shape = t.logical_shape();
    const int rank = static_cast<int>(shape.rank());
    const int row_dim = rank - 2;
    uint32_t rows = 1U;
    for (int dim = 0; dim <= row_dim; ++dim) {
        rows *= shape[dim];
    }
    // ttnn::sum over {0, ..., rank - 2} reduces the dims before S one by one. For bf16 each gets its own
    // fast_reduce_nc pass, and a pass over a dim of size 1 still writes a full-size fp32 copy. For fp32 each costs a
    // transpose, a reduction and a transpose back. When S is a whole number of tiles, the dims before S fold into it
    // without moving data, because [..., S, C] and [1, ..., 1, rows, C] have the same tile order, and a single
    // reduction over the rows is left. Otherwise only the dims larger than 1 are reduced.
    ttnn::Tensor input = t;
    ttsl::SmallVector<int> dims;
    if (row_dim >= 0 && rows != shape[row_dim] && shape[row_dim] % tt::constants::TILE_HEIGHT == 0U) {
        ttsl::SmallVector<uint32_t> folded(rank, 1U);
        folded[row_dim] = rows;
        folded[rank - 1] = shape[-1];
        input = ttnn::reshape(t, ttnn::Shape(folded));
        dims.push_back(row_dim);
    } else {
        for (int dim = 0; dim <= row_dim; ++dim) {
            if (shape[dim] != 1U) {
                dims.push_back(dim);
            }
        }
    }
    // An empty dim list would make ttnn::sum reduce over every dim, the last one included.
    if (dims.empty()) {
        return t;
    }
    return ttnn::sum(
        input,
        dims,
        /* keep_dim */ true,
        /* output_mem_config */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::precise());
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
