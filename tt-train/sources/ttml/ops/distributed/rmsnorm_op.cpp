// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_op.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <stdexcept>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "metal/operations.hpp"
#include "ops/rmsnorm_op.hpp"
#include "ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_post_all_gather.hpp"
#include "ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

namespace {

// Largest local width, in elements, whose full row fits the interleaved forward programs this wrapper calls.
// Both keep that row in L1: pre_all_gather always, and post_all_gather until one buffer exceeds 95% of L1,
// which is past the width where the sum of the full-row buffers already overflows.
//
// pre (single-buffered x^2 fallback in LayerNormPreAllGatherProgramFactory, fp32 dest acc, fp32 stats, no residual):
//   2·Wt bf16 input tiles + Wt fp32 x^2 tiles + one fp32 scaler + one fp32 output.
// post (LayerNormPostAllGatherProgramFactory, RMSNorm, gamma, no beta, bf16 input and output, fp32 stats):
//   Wt bf16 input + Wt bf16 gamma + Wt fp32 x_normed + Wt bf16 output
//   + num_shards fp32 stats tiles + one bf16 epsilon + one bf16 scaler + one fp32 variance + one fp32 rsqrt.
uint32_t max_local_width_fitting_forward(tt::tt_metal::IDevice& device, uint32_t num_shards) {
    const uint32_t l1_budget = static_cast<uint32_t>(
        device.l1_size_per_core() - device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    const uint32_t bf16_tile = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t fp32_tile = tt::tile_size(tt::DataFormat::Float32);

    const uint32_t pre_fixed = 2U * fp32_tile;
    const uint32_t pre_per_tile = 2U * bf16_tile + fp32_tile;
    const uint32_t post_fixed = num_shards * fp32_tile + 2U * bf16_tile + 2U * fp32_tile;
    const uint32_t post_per_tile = 3U * bf16_tile + fp32_tile;

    auto max_tiles = [](uint32_t budget, uint32_t fixed, uint32_t per_tile) -> uint32_t {
        if (budget <= fixed || per_tile == 0U) {
            return 0U;
        }
        return (budget - fixed) / per_tile;
    };
    const uint32_t max_wt =
        std::min(max_tiles(l1_budget, pre_fixed, pre_per_tile), max_tiles(l1_budget, post_fixed, post_per_tile));
    return max_wt * tt::constants::TILE_WIDTH;
}

}  // namespace

// The non-zero columns of the gathered stats are one sum(x^2) per shard, so one sum over the last dim scaled
// by 1/C is the mean square.
ttnn::Tensor rms_from_gathered_stats(const ttnn::Tensor& stats, uint32_t full_width, float epsilon) {
    auto mean_square = ttnn::sum(
        stats,
        /* dim_arg */ -1,
        /* keepdim */ true,
        /* memory_config_arg */ std::nullopt,
        core::ComputeKernelConfig::precise(),
        /* scalar */ 1.F / static_cast<float>(full_width));
    auto rms = ttnn::sqrt(ttnn::add(mean_square, epsilon));
    return ttnn::typecast(rms, ttnn::DataType::BFLOAT16);
}

autograd::TensorPtr rmsnorm(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& gamma,
    float epsilon,
    std::optional<uint32_t> cluster_axis) {
    const auto& x_shape = tensor->get_value().logical_shape();
    if (x_shape.rank() != 4) {
        throw std::invalid_argument("distributed rmsnorm only supports rank-4 input tensors.");
    }
    const uint32_t local_width = x_shape[-1];
    const auto& g_shape = gamma->get_value().logical_shape();
    if (g_shape.rank() != 4 || g_shape[0] != 1 || g_shape[1] != 1 || g_shape[2] != 1 || g_shape[3] != local_width) {
        throw std::invalid_argument(fmt::format(
            "distributed rmsnorm: gamma must be [1, 1, 1, {}] (sharded like the input), got {}", local_width, g_shape));
    }

    // Number of devices the hidden dim is sharded over.
    const uint32_t num_shards = ttnn_fixed::distributed::get_axis_size(cluster_axis);
    if (num_shards == 1U) {
        // The local row is the full row: the fused single-device kernel needs no stats exchange.
        return ops::rmsnorm(tensor, gamma, epsilon);
    }

    if (local_width % tt::constants::TILE_WIDTH != 0) {
        throw std::invalid_argument(fmt::format(
            "distributed rmsnorm: local hidden size {} must be a multiple of {}",
            local_width,
            tt::constants::TILE_WIDTH));
    }
    const uint32_t full_width = local_width * num_shards;

    const auto compute_kernel_config = core::ComputeKernelConfig::precise();
    const auto& x = tensor->get_value();
    const uint32_t max_local_width = max_local_width_fitting_forward(*x.device(), num_shards);
    if (local_width > max_local_width) {
        throw std::invalid_argument(fmt::format(
            "distributed rmsnorm: local hidden size {} does not fit the forward's whole-row L1 programs "
            "(rms_norm_pre_all_gather / rms_norm_post_all_gather). The maximum local width on this device is {}.",
            local_width,
            max_local_width));
    }

    // Stats stay fp32 end to end: a bf16 sum of x^2 over a long row keeps only ~3 significant digits.
    auto local_stats = ttnn::rms_norm_pre_all_gather(
        x, ttnn::DataType::FLOAT32, /* residual_input_tensor */ std::nullopt, compute_kernel_config);
    auto stats = ttnn_fixed::distributed::all_gather(local_stats, /* dim */ 3, cluster_axis);

    auto out = autograd::create_tensor(ttnn::rms_norm_post_all_gather(
        x,
        stats,
        epsilon,
        gamma->get_value(),
        /* bias */ std::nullopt,
        /* memory_config */ std::nullopt,
        compute_kernel_config,
        /* program_config */ std::nullopt,
        /* dtype */ x.dtype()));

    const bool inputs_require_grad = autograd::any_requires_grad(tensor, gamma);
    if (autograd::ctx().get_gradient_mode() == autograd::GradMode::DISABLED || !inputs_require_grad) {
        out->set_requires_grad(inputs_require_grad);
        return out;
    }

    // Backward needs only rms, one value per row, so the gathered stats are released here.
    auto rms = rms_from_gathered_stats(stats, full_width, epsilon);

    autograd::GradFunction grad = [tensor, gamma, out, rms, full_width, cluster_axis]() {
        const auto& x = tensor->get_value();
        const auto& g = gamma->get_value();
        const auto& dL_dout = out->get_grad();

        auto local_sums = metal::rmsnorm_bw_local_sum(x, g, dL_dout);
        auto gathered_sums = ttnn_fixed::distributed::all_gather(local_sums, /* dim */ 3, cluster_axis);

        auto grads = metal::rmsnorm_bw_external_scale(x, g, rms, dL_dout, gathered_sums, full_width);
        if (grads.size() != 2U) {
            throw std::runtime_error(fmt::format(
                "rmsnorm_bw_external_scale returned an unexpected number of gradients. Expected 2, got {}",
                grads.size()));
        }
        if (grads[0].has_value()) {
            tensor->add_grad(grads[0].value());
        }
        if (grads[1].has_value()) {
            gamma->add_grad(grads[1].value());
        }
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor, gamma));
    return out;
}

}  // namespace ttml::ops::distributed
