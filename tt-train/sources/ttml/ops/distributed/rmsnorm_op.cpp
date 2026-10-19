// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_op.hpp"

#include <fmt/format.h>

#include <stdexcept>
#include <tt-metalium/constants.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ops/rmsnorm_op.hpp"
#include "ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_post_all_gather.hpp"
#include "ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

namespace {

// rms of the full row from the gathered rms_norm_pre_all_gather stats [B,1,S,32*num_shards]. Each shard's tile
// holds its sum(x^2) in column 0 and undefined values elsewhere, so a matmul with a selector that is 1/C at row
// 32*k of column 0 picks out exactly those columns and averages them: [B,1,S,32] with the mean square in column 0.
ttnn::Tensor rms_from_gathered_stats(
    const ttnn::Tensor& stats, uint32_t num_shards, uint32_t full_width, float epsilon) {
    constexpr uint32_t tile_width = tt::constants::TILE_WIDTH;
    std::vector<float> selector(num_shards * tile_width * tile_width, 0.F);
    for (uint32_t shard = 0; shard < num_shards; ++shard) {
        selector[shard * tile_width * tile_width] = 1.F / static_cast<float>(full_width);
    }
    auto selector_tensor = core::from_vector<float, ttnn::DataType::FLOAT32>(
        selector, ttnn::Shape({1U, 1U, num_shards * tile_width, tile_width}), &autograd::ctx().get_device());

    auto mean_square = ttnn::matmul(
        stats,
        selector_tensor,
        /* transpose_a */ false,
        /* transpose_b */ false,
        /* memory_config */ std::nullopt,
        /* dtype */ ttnn::DataType::FLOAT32,
        /* program_config */ std::nullopt,
        /* activation */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::precise());
    auto rms = ttnn::sqrt(ttnn::add(mean_square, epsilon));
    return ttnn::typecast(rms, ttnn::DataType::BFLOAT16);
}

}  // namespace

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

    // Backward needs only rms, one value per row, so the gathered stats are released here.
    auto rms = rms_from_gathered_stats(stats, num_shards, full_width, epsilon);

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
