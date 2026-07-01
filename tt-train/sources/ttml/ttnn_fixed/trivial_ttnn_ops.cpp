// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "trivial_ttnn_ops.hpp"

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/moreh/moreh_mean/moreh_mean.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/rand/rand.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

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

    auto log_t_sum_over_dim = ttnn::log(t_sum_over_dim, /*fast_and_approximate_mode=*/true);
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
    auto inv_b = ttnn::reciprocal(b);
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

tt::tt_metal::Tensor sample(
    const tt::tt_metal::Tensor& t,
    float temperature,
    uint32_t seed,
    std::optional<tt::tt_metal::Tensor> logits_padding_mask) {
    auto* device = &ttml::autograd::ctx().get_device();

    ttnn::Tensor out = t;

    if (temperature > 0.0F) {
        namespace distributed = tt::tt_metal::distributed;

        // NOTE on seed==0: we deliberately pass it through to ttnn::rand verbatim to honor that
        // op's documented contract (seed==0 => host time-based entropy, non-reproducible). The
        // tradeoff: ttnn::rand applies the per-device seed offset (see below) ONLY on the seed!=0
        // path, so with seed==0 each device's noise comes from an independent host-entropy draw --
        // neither reproducible nor *guaranteed* distinct across devices (divergence then depends on
        // host RNG ordering, not on the sharded mesh index). Callers that need reproducible,
        // guaranteed-per-device-distinct sampling (e.g. GRPO training) MUST pass a nonzero seed.

        // The logits tensor `t` is the per-device LOCAL tensor, so out.logical_shape() is the
        // local shape [B_local, 1, new_tokens, padded_V]. On a multi-device (DDP) mesh this tensor
        // is effectively replicated, and ttnn::rand with no mesh_mapper produces IDENTICAL RNG on
        // every device (same seed + same per-core offset), giving identical Gumbel noise and thus
        // identical argmax samples. To make each device draw DISTINCT but reproducible noise, we
        // pass a MeshMapperConfig with a Shard placement on the data-parallel mesh axis. ttnn::rand
        // then sets a per-device seed offset = (sharded linear mesh index) * num_cores, so each
        // device gets a disjoint LFSR stream that is deterministic given `seed`.
        const auto mesh_shape = ttml::autograd::ctx().get_mesh_shape();
        const auto local_shape = out.logical_shape();

        // Identify the non-trivial (data-parallel) mesh axis: the first dim with size > 1.
        std::optional<size_t> shard_axis;
        for (size_t i = 0; i < mesh_shape.dims(); ++i) {
            if (mesh_shape[i] > 1U) {
                shard_axis = i;
                break;
            }
        }

        ttnn::Tensor rand;
        if (shard_axis.has_value()) {
            // compute_shard_shape() inside ttnn::rand DIVIDES shape[shard_dim] by the mesh axis
            // size, and TT_FATALs if it is not divisible. To keep the PER-DEVICE rand shape exactly
            // equal to the local logits shape (required for the elementwise ttnn::add below), we
            // pass a GLOBAL shape whose tensor batch dim (dim 0, the data-parallel dim) is the local
            // batch multiplied by the mesh axis size. After division it returns to the local size,
            // so divisibility always holds even when B_local == 1.
            constexpr int kShardTensorDim = 0;  // batch dim is the data-parallel dim
            const uint32_t mesh_axis_size = mesh_shape[*shard_axis];

            ttnn::Shape::Container global_dims(local_shape.view().begin(), local_shape.view().end());
            global_dims[kShardTensorDim] *= mesh_axis_size;
            const ttnn::Shape global_shape(std::move(global_dims));

            // placements size must match mesh dims: Shard the data-parallel axis, Replicate the rest.
            distributed::MeshMapperConfig mapper;
            mapper.placements.reserve(mesh_shape.dims());
            for (size_t i = 0; i < mesh_shape.dims(); ++i) {
                if (i == *shard_axis) {
                    mapper.placements.push_back(distributed::MeshMapperConfig::Shard{kShardTensorDim});
                } else {
                    mapper.placements.push_back(distributed::MeshMapperConfig::Replicate{});
                }
            }

            rand = ttnn::rand(
                /* size */ global_shape,
                /* device */ *device,
                /* dtype */ out.dtype(),
                /* layout */ out.layout(),
                /* memory_config */ ttnn::types::DRAM_MEMORY_CONFIG,
                /* from */ 0.00001F,
                /* to */ 0.99F,
                /* seed */ seed,
                /* mesh_mapper */ mapper);
        } else {
            // Single-device (or all-unit) mesh: no data-parallel axis, distinct per-device noise is
            // not needed. Draw directly on the local shape with no mapper.
            rand = ttnn::rand(
                /* size */ local_shape,
                /* device */ *device,
                /* dtype */ out.dtype(),
                /* layout */ out.layout(),
                /* memory_config */ ttnn::types::DRAM_MEMORY_CONFIG,
                /* from */ 0.00001F,
                /* to */ 0.99F,
                /* seed */ seed);
        }

        // Gumbel sampling trick: -log(-log(U)), where U ~ Uniform(0, 1)
        // See: https://en.wikipedia.org/wiki/Gumbel_distribution#Random_variate_generation
        rand = ttnn::neg(ttnn::log(ttnn::neg(ttnn::log(rand))));
        out = ttnn::mul_sfpu(out, 1.0F / temperature);
        out = ttnn::add(out, rand);
    }

    if (logits_padding_mask.has_value()) {
        // subtract a large number from the logits where the padding mask is set
        out = ttnn::subtract(out, logits_padding_mask.value());
    }

    return ttnn::argmax(ttnn::untilize(out), 3, true);
}

tt::tt_metal::Tensor to_l1_interleaved(const tt::tt_metal::Tensor& t) {
    return ttnn::to_memory_config(t, ttnn::L1_MEMORY_CONFIG);
}

tt::tt_metal::Tensor to_dram_interleaved(const tt::tt_metal::Tensor& t) {
    return ttnn::to_memory_config(t, ttnn::DRAM_MEMORY_CONFIG);
}

}  // namespace ttml::ttnn_fixed
