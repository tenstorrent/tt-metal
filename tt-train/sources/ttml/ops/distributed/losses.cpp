// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "losses.hpp"

#include <fmt/core.h>

#include <enchantum/enchantum.hpp>
#include <stdexcept>
#include <variant>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "metal/ops/select_target_logit/device/select_target_logit_device_operation.hpp"
#include "metal/ops/subtract_at_target/device/subtract_at_target_device_operation.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_configs.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace {

// Map flat device index to TP rank along cluster_axis (row-major mesh enumeration).
uint32_t tp_rank_from_device_idx(
    size_t idx, const tt::tt_metal::distributed::MeshShape& mesh_shape, std::optional<uint32_t> cluster_axis) {
    if (!cluster_axis.has_value() || mesh_shape.dims() <= 1U) {
        return static_cast<uint32_t>(idx);
    }
    const uint32_t dim1 = mesh_shape[1U];
    return (cluster_axis.value() == 0U) ? static_cast<uint32_t>(idx) / dim1 : static_cast<uint32_t>(idx) % dim1;
}

}  // namespace

namespace ttml::ops::distributed {

autograd::TensorPtr vocab_parallel_cross_entropy_loss(
    const autograd::TensorPtr& logits, const autograd::TensorPtr& targets, std::optional<uint32_t> cluster_axis) {
    auto* device = &autograd::ctx().get_device();

    const auto logits_shape = logits->get_value().logical_shape();
    const auto targets_shape = targets->get_value().logical_shape();

    if (logits->get_value().dtype() != ttnn::DataType::BFLOAT16) {
        throw std::logic_error(fmt::format(
            "vocab_parallel_cross_entropy_loss: logits must be BFLOAT16, got {}",
            enchantum::to_string(logits->get_value().dtype())));
    }
    if (logits_shape.rank() != 4U) {
        throw std::logic_error(
            fmt::format("vocab_parallel_cross_entropy_loss: logits must be rank 4, got shape {}", logits_shape));
    }
    if (targets_shape.rank() != 2U && targets_shape.rank() != 4U) {
        throw std::logic_error(
            fmt::format("vocab_parallel_cross_entropy_loss: targets must be rank 2 or 4, got shape {}", targets_shape));
    }
    if (logits_shape[0] != targets_shape[0]) {
        throw std::logic_error(fmt::format(
            "vocab_parallel_cross_entropy_loss: batch dimension (dim 0) must match.\n"
            "Got: logits shape {}, targets shape {}",
            logits_shape,
            targets_shape));
    }
    const uint32_t target_seq = (targets_shape.rank() == 2U) ? targets_shape[1] : targets_shape[2];
    if (logits_shape[2] != target_seq) {
        throw std::logic_error(fmt::format(
            "vocab_parallel_cross_entropy_loss: sequence dimension must match.\n"
            "Got: logits shape {} (S={}), targets shape {} (S={})",
            logits_shape,
            logits_shape[2],
            targets_shape,
            target_seq));
    }

    // Contract: logits must be sharded (not replicated) across the TP cluster axis.
    // If they're replicated (e.g. because an upstream ColumnParallelLinear was constructed
    // with gather_output=true), every device's local_sum_exp ends up identical and the
    // all_reduce(SUM) below would silently inflate the loss by log(tp_size).  Catch that
    // misuse here with an explicit error.
    //
    // Note: we deliberately do *not* assert that the recorded Shard.dim equals 3.  The
    // topology placement records the dim in the *source* tensor's frame; for a
    // ColumnParallelLinear the source is the weight ([1,1,out,in], sharded at dim 2) and
    // the matmul does not relabel that to the output's last dim.  So a correctly
    // vocab-sharded LM head output may legitimately report Shard{dim=2}.  The op only
    // relies on logits_shape[3] equalling the per-device V/tp_size, which is true in
    // either labelling.
    {
        const auto& topology = logits->get_value().tensor_topology();
        const auto& placements = topology.placements();
        const uint32_t tp_axis_idx = cluster_axis.value_or(0U);
        if (tp_axis_idx >= placements.size()) {
            throw std::logic_error(fmt::format(
                "vocab_parallel_cross_entropy_loss: cluster_axis={} is out of range for tensor topology "
                "with {} placement(s).  Expected logits to be sharded across the TP cluster axis.",
                tp_axis_idx,
                placements.size()));
        }
        const auto& tp_placement = placements[tp_axis_idx];
        if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Replicate>(tp_placement)) {
            throw std::logic_error(
                "vocab_parallel_cross_entropy_loss: logits are replicated along the TP cluster axis, but "
                "this op requires them to be vocab-sharded ([B,1,S,V/tp_size] per device).  If "
                "you are using ColumnParallelLinear at the LM head, construct it with "
                "gather_output=false.  Otherwise the loss would be silently inflated by "
                "log(tp_size).");
        }
    }

    const uint32_t B = logits_shape[0];
    const uint32_t S = logits_shape[2];
    const uint32_t local_V = logits_shape[3];
    const uint32_t N = B * S;

    auto mesh_shape = device->shape();

    // Step 1: local max [B,1,S,1] per device (BF16 — no precision loss)
    auto local_max = ttnn::max(logits->get_value(), 3, /* keepdim */ true);

    // Step 2: all-gather local maxes → [B,1,S,tp_size] → global max [B,1,S,1]
    auto all_max_val = ttnn_fixed::distributed::all_gather(local_max, 3, cluster_axis);
    auto global_max = ttnn::max(all_max_val, 3, /* keepdim */ true);

    // Step 3: shifted exp, local sum → all-reduce → global sum [B,1,S,1]
    // BF16 inputs are read and accumulated into an FP32 destination.
    auto shifted = ttnn::subtract(logits->get_value(), global_max, ttnn::DataType::FLOAT32);
    auto local_exp = ttnn::exp(shifted);
    auto local_sum = ttnn::sum(local_exp, 3, /* keepdim */ true, std::nullopt, core::ComputeKernelConfig::precise());
    auto global_sum = ttnn_fixed::distributed::all_reduce(local_sum, cluster_axis);

    // log_normalizer = global_max + log(global_sum)  [B,1,S,1]
    // global_max is BF16; log(global_sum) is FP32 — output_dtype promotes the add.
    auto log_normalizer = ttnn::add(global_max, ttnn::log(global_sum), ttnn::DataType::FLOAT32);

    // Step 4: target_logit_local via select_target_logit:
    // Each TP device k owns vocab [k*local_V, (k+1)*local_V).  We pre-allocate
    // a [B,1,S,1] BF16 output replicated across all devices.
    // Then for each device we call ttml_select_target_logit with its shard boundaries.
    auto targets_raw = targets->get_value();
    if (targets_raw.layout() != ttnn::Layout::ROW_MAJOR) {
        targets_raw = ttnn::to_layout(targets_raw, ttnn::Layout::ROW_MAJOR);
    }
    if (targets_raw.logical_shape().rank() != 2U) {
        targets_raw = ttnn::reshape(targets_raw, ttnn::Shape({B, S}));
    }
    const auto& logits_val = logits->get_value();

    // Can use empty because ttml_select_target_logit zero-initializes its output tile before writing.
    auto gather_output = ttnn::empty(
        ttnn::Shape({B, 1U, S, 1U}), ttnn::DataType::BFLOAT16, ttnn::Layout::TILE, device, ttnn::DRAM_MEMORY_CONFIG);

    auto logit_shards = ttnn::distributed::get_device_tensors(logits_val);
    auto target_shards = ttnn::distributed::get_device_tensors(targets_raw);
    auto output_shards = ttnn::distributed::get_device_tensors(gather_output);

    // For each TP shard k, extract target_logit_local[b,s] = logits[b,s,targets[b,s]]
    // if targets[b,s] ∈ [k*local_V, (k+1)*local_V), else 0.
    // After all-reduce the contributions sum to the full x[b,s,targets[b,s]].
    std::vector<uint32_t> tp_ranks(logit_shards.size());
    for (size_t i = 0; i < logit_shards.size(); ++i) {
        tp_ranks[i] = tp_rank_from_device_idx(i, mesh_shape, cluster_axis);
        ttnn::prim::ttml_select_target_logit(
            logit_shards[i], target_shards[i], tp_ranks[i] * local_V, (tp_ranks[i] + 1U) * local_V, output_shards[i]);
    }
    // All-reduce to collect contributions from all TP shards  [B,1,S,1]
    auto target_logit = ttnn_fixed::distributed::all_reduce(gather_output, cluster_axis);

    // Step 5: per-position loss = log_normalizer − target_logit  →  mean over B*S
    // log_normalizer is FP32, target_logit is BF16 — output_dtype keeps subtraction in FP32.
    auto per_pos = ttnn::subtract(log_normalizer, target_logit, ttnn::DataType::FLOAT32);
    auto s0 = ttnn::sum(per_pos, 0, /* keepdim */ true, std::nullopt, core::ComputeKernelConfig::precise());
    auto s2 = ttnn::sum(s0, 2, /* keepdim */ true, std::nullopt, core::ComputeKernelConfig::precise());
    // Transition back to BF16 to match the non-sharded cross_entropy_loss output type.
    auto loss_val = ttnn::multiply(s2, 1.0F / static_cast<float>(N), ttnn::DataType::BFLOAT16);

    auto out = autograd::create_tensor(loss_val);

    // ---------------------------------------------------------------
    // Backward  (purely local — no inter-device communication)
    //
    // dCE/dx_k = (softmax_k / N − onehot_k / N) * grad_output
    //
    // Like the reference cross_entropy_bw kernel, we recompute local_exp
    // from logits + global_max instead of saving it from the forward
    // pass.  This trades compute for memory: we avoid keeping a full
    // [B,1,S,local_V] FP32 tensor alive across the forward-backward
    // boundary.  global_sum is captured from the forward pass — its
    // shape is only [B,1,S,1] so the memory cost is negligible, and
    // reusing it removes the need for an all_reduce in the backward.
    //
    // We scale by 1/N first, cast to BF16, then scatter-subtract 1/N at the
    // target position; the ordering matches the
    // reference cross_entropy_bw kernel, whose writer does the onehot subtraction
    // directly on the already-scaled BF16 tile.
    // ---------------------------------------------------------------
    autograd::GradFunction grad_fn =
        [logits, out, global_max, global_sum, targets_raw, local_V, tp_ranks = std::move(tp_ranks), N]() {
            if (!out->is_grad_initialized()) {
                return;
            }
            const float inv_N = 1.0F / static_cast<float>(N);

            // Recompute local_exp from logits + global_max (captured from forward).
            // global_sum is also captured from the forward pass — its footprint is [B,1,S,1]
            // so keeping it alive across fwd/bwd is negligible, and this lets the backward
            // stay purely local (no inter-device communication).
            auto shifted = ttnn::subtract(logits->get_value(), global_max, ttnn::DataType::FLOAT32);
            auto local_exp = ttnn::exp(shifted);

            auto softmax_k = ttnn::multiply(local_exp, ttnn::reciprocal(global_sum));
            auto scaled_softmax = ttnn::multiply(softmax_k, inv_N, ttnn::DataType::BFLOAT16);
            auto sm_shards = ttnn::distributed::get_device_tensors(scaled_softmax);
            auto tgt_shards = ttnn::distributed::get_device_tensors(targets_raw);

            for (size_t i = 0; i < sm_shards.size(); ++i) {
                ttnn::prim::ttml_subtract_at_target(
                    sm_shards[i],
                    tgt_shards[i],
                    tp_ranks[i] * local_V,
                    (tp_ranks[i] + 1U) * local_V,
                    sm_shards[i],
                    inv_N);
            }

            logits->add_grad(ttnn::multiply(scaled_softmax, out->get_grad(), ttnn::DataType::BFLOAT16));
        };

    out->set_node(autograd::add_backward_node(std::move(grad_fn), out, logits));
    return out;
}

}  // namespace ttml::ops::distributed
