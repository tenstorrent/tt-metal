// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_op.hpp"

#include <fmt/format.h>

#include <stdexcept>
#include <tt-metalium/constants.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "ops/rmsnorm_op.hpp"
#include "ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_post_all_gather.hpp"
#include "ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_post_all_gather_bw.hpp"
#include "ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.hpp"
#include "ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather_bw.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

namespace {

// Number of devices the hidden dim is sharded over. Without an axis, all_gather walks every device
// as one line, which is only a real fabric line on a 1-D mesh.
uint32_t get_num_shards(const std::optional<uint32_t> cluster_axis) {
    auto* device = &autograd::ctx().get_device();
    const auto& mesh_shape = device->shape();
    if (cluster_axis.has_value()) {
        if (cluster_axis.value() >= mesh_shape.dims()) {
            throw std::invalid_argument(fmt::format(
                "distributed rmsnorm: cluster_axis {} is out of range for mesh shape {}",
                cluster_axis.value(),
                mesh_shape));
        }
        return mesh_shape[cluster_axis.value()];
    }
    if (!mesh_shape.is_line_topology()) {
        throw std::invalid_argument(fmt::format(
            "distributed rmsnorm: cluster_axis must be specified on a multi-dimensional mesh {}", mesh_shape));
    }
    return static_cast<uint32_t>(device->num_devices());
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

    const uint32_t num_shards = get_num_shards(cluster_axis);
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

    const auto compute_kernel_config = core::ComputeKernelConfig::precise();
    const auto& x = tensor->get_value();

    // Stats stay fp32 end to end: a bf16 sum of x^2 over a long row keeps only ~3 significant digits,
    // and the backward reuses it.
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

    autograd::GradFunction grad = [tensor, gamma, out, stats, epsilon, cluster_axis, compute_kernel_config]() {
        const auto& x = tensor->get_value();
        const auto& g = gamma->get_value();
        const auto& dL_dout = out->get_grad();

        auto local_bw_stats = ttnn::rms_norm_pre_all_gather_bw(
            x, dL_dout, stats, epsilon, g, /* memory_config */ std::nullopt, compute_kernel_config);
        auto bw_stats = ttnn_fixed::distributed::all_gather(local_bw_stats, /* dim */ 3, cluster_axis);

        auto grads = ttnn::rms_norm_post_all_gather_bw(
            x, dL_dout, stats, bw_stats, epsilon, g, /* memory_config */ std::nullopt, compute_kernel_config);
        if (grads.size() != 2U) {
            throw std::runtime_error(fmt::format(
                "rms_norm_post_all_gather_bw returned an unexpected number of gradients. Expected 2, got {}",
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
