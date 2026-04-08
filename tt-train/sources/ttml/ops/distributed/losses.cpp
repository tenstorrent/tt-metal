// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "losses.hpp"

#include <stdexcept>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/distributed/comm_ops.hpp"

namespace ttml::ops::distributed {

autograd::TensorPtr sharded_cross_entropy_loss(
    const autograd::TensorPtr& logits, const autograd::TensorPtr& targets, std::optional<uint32_t> cluster_axis) {
    auto* device = &autograd::ctx().get_device();

    const auto logits_shape = logits->get_value().logical_shape();
    if (logits_shape.rank() != 4U) {
        throw std::logic_error(
            fmt::format("sharded_cross_entropy_loss: logits must be rank 4, got shape {}", logits_shape));
    }
    const uint32_t B = logits_shape[0];
    const uint32_t S = logits_shape[2];
    const uint32_t local_V = logits_shape[3];
    const uint32_t N = B * S;

    uint32_t tp_size = 1U;
    auto mesh_shape = device->shape();
    if (cluster_axis.has_value() && mesh_shape.dims() == 2) {
        tp_size = mesh_shape[cluster_axis.value()];
    } else {
        tp_size = static_cast<uint32_t>(device->num_devices());
    }

    // ---------------------------------------------------------------
    // Forward  (all computation in float32 for numerical stability)
    // ---------------------------------------------------------------

    // Cast logits to float32
    auto x = ttnn::typecast(logits->get_value(), ttnn::DataType::FLOAT32);

    // Step 1: local max [B,1,S,1]
    auto local_max = ttnn::max(x, 3, /* keepdim */ true);

    // Step 2: all-gather local maxes → [B,1,S,tp_size] → global max [B,1,S,1]
    auto lm_tensor = autograd::create_tensor(local_max, /* requires_grad */ false);
    auto all_max_val = all_gather(lm_tensor, /* dim */ 3, cluster_axis)->get_value();
    auto global_max = ttnn::max(all_max_val, 3, /* keepdim */ true);

    // Step 3: shifted exp, local sum → all-reduce → global sum [B,1,S,1]
    auto shifted = ttnn::subtract(x, global_max);
    auto local_exp = ttnn::exp(shifted);
    auto local_sum = ttnn::sum(local_exp, 3, /* keepdim */ true, std::nullopt, core::ComputeKernelConfig::precise());
    auto ls_tensor = autograd::create_tensor(local_sum, /* requires_grad */ false);
    auto global_sum = all_reduce(ls_tensor, /* noop_backward */ true, cluster_axis)->get_value();

    // log_normalizer = global_max + log(global_sum)  [B,1,S,1]
    auto log_normalizer = ttnn::add(global_max, ttnn::log(global_sum));

    // Step 4: on-device one-hot for target logit extraction.
    //
    // Build a vocab-range tensor on CPU: shape [tp_size, 1, 1, local_V] where
    // row k holds the global vocab indices for TP rank k.  Shard to each device
    // so that device k sees exactly [1,1,1,local_V] = [k*local_V .. (k+1)*local_V).
    // Tile-padding zeros (if local_V is not tile-aligned) cannot match any valid
    // target index because the padded logit entries are also zero, so they
    // contribute nothing to the sum.
    std::vector<float> vocab_range_cpu(static_cast<size_t>(tp_size) * static_cast<size_t>(local_V));
    for (uint32_t k = 0; k < tp_size; ++k) {
        for (uint32_t v = 0; v < local_V; ++v) {
            vocab_range_cpu[static_cast<size_t>(k) * static_cast<size_t>(local_V) + static_cast<size_t>(v)] =
                static_cast<float>(k * local_V + v);
        }
    }
    const auto vocab_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0U, cluster_axis);
    auto vocab_range = core::from_vector<float, ttnn::DataType::FLOAT32>(
        vocab_range_cpu, ttnn::Shape({tp_size, 1U, 1U, local_V}), device, ttnn::Layout::TILE, vocab_mapper.get());

    // targets_f32: ensure [B,1,S,1] float32 for broadcast comparison
    auto targets_val = targets->get_value();
    if (targets_val.logical_shape().rank() == 2U) {
        targets_val = ttnn::reshape(targets_val, ttnn::Shape({B, 1U, S, 1U}));
    }
    auto targets_f32 = ttnn::typecast(targets_val, ttnn::DataType::FLOAT32);

    // one_hot: [B,1,S,1] == [1,1,1,local_V]  →  [B,1,S,local_V]  (broadcast)
    auto one_hot = ttnn::typecast(ttnn::eq(targets_f32, vocab_range), ttnn::DataType::FLOAT32);

    // target_logit_local = sum(x * one_hot, dim=3)  [B,1,S,1]
    auto target_logit_local = ttnn::sum(
        ttnn::multiply(x, one_hot), 3, /* keepdim */ true, std::nullopt, core::ComputeKernelConfig::precise());

    // All-reduce to collect contributions from all TP shards  [B,1,S,1]
    auto tl_tensor = autograd::create_tensor(target_logit_local, /* requires_grad */ false);
    auto target_logit = all_reduce(tl_tensor, /* noop_backward */ true, cluster_axis)->get_value();

    // Step 5: per-position loss = log_normalizer − target_logit  →  mean over B*S
    auto per_pos = ttnn::subtract(log_normalizer, target_logit);
    auto s0 = ttnn::sum(per_pos, 0, /* keepdim */ true, std::nullopt, core::ComputeKernelConfig::precise());
    auto s2 = ttnn::sum(s0, 2, /* keepdim */ true, std::nullopt, core::ComputeKernelConfig::precise());
    auto loss_val = ttnn::multiply(s2, 1.0F / static_cast<float>(N));

    auto out = autograd::create_tensor(loss_val);

    // ---------------------------------------------------------------
    // Backward  (purely local — no inter-device communication)
    //
    // dCE/dx_k = (softmax_k − onehot_k) / N * grad_output
    // softmax_k = exp(shifted_k) / global_sum
    // ---------------------------------------------------------------
    autograd::GradFunction grad_fn = [logits, out, shifted, global_sum, one_hot, N]() {
        if (!out->is_grad_initialized()) {
            return;
        }
        auto grad_out = ttnn::typecast(out->get_grad(), ttnn::DataType::FLOAT32);
        auto softmax_k = ttnn::multiply(ttnn::exp(shifted), ttnn::reciprocal(global_sum));
        auto diff = ttnn::subtract(softmax_k, one_hot);
        auto grad = ttnn::multiply(ttnn::multiply(diff, 1.0F / static_cast<float>(N)), grad_out);
        logits->add_grad(ttnn::typecast(grad, ttnn::DataType::BFLOAT16));
    };

    out->set_node(autograd::add_backward_node(std::move(grad_fn), out, logits));
    return out;
}

}  // namespace ttml::ops::distributed
