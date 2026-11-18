// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "losses.hpp"

#include <fmt/core.h>

#include <enchantum/enchantum.hpp>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "metal/ops/vocab_parallel_cross_entropy/vocab_parallel_cross_entropy.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

autograd::TensorPtr vocab_parallel_cross_entropy_loss(
    const autograd::TensorPtr& logits,
    const autograd::TensorPtr& targets,
    std::optional<uint32_t> cluster_axis,
    ReduceType reduce) {
    if (reduce != ReduceType::NONE && reduce != ReduceType::MEAN) {
        throw std::logic_error(fmt::format(
            "vocab_parallel_cross_entropy_loss: only NONE and MEAN reductions are supported. Got: {}",
            enchantum::to_string(reduce)));
    }

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

    const uint32_t B = logits_shape[0];
    const uint32_t S = logits_shape[2];
    const uint32_t N = B * S;
    // For MEAN, 1/N is folded into the gradient's row scale and the upstream scalar multiplies through.
    // For NONE the upstream gradient is the per-position [B,1,S,1] tensor itself.
    const float inv_N = (reduce == ReduceType::MEAN) ? (1.0F / static_cast<float>(N)) : 1.0F;

    auto targets_raw = targets->get_value();
    if (targets_raw.layout() != ttnn::Layout::ROW_MAJOR) {
        targets_raw = ttnn::to_layout(targets_raw, ttnn::Layout::ROW_MAJOR);
    }
    if (targets_raw.logical_shape().rank() != 2U) {
        targets_raw = ttnn::reshape(targets_raw, ttnn::Shape({B, S}));
    }

    // Collapse-resistant statistics (arXiv:2610.31642, Section 2; see metal/ops/vocab_parallel_cross_entropy/
    // vocab_parallel_cross_entropy.hpp). Each shard reduces its columns to (maximum, global first-maximum
    // column, tail without that element, target logit if owned) in FP32; the records are all-gathered across
    // TP, which is exact, and every device combines them identically: one global first maximum, the tail
    // without exactly that element, log1p in FP32, loss = -((x_t - M) - log1p(T)).
    auto stats = metal::vocab_parallel_ce_stats(logits->get_value(), targets_raw, cluster_axis);
    auto gathered = ttnn_fixed::distributed::all_gather(stats, 3, cluster_axis);
    stats.deallocate();
    auto combined = metal::vocab_parallel_ce_combine(gathered);
    auto per_pos = combined[0];
    auto state = combined[1];
    gathered.deallocate();

    // per_pos [B,1,S,1] FP32. MEAN keeps FP32 for the precise sum reductions; NONE matches the BF16 output of
    // the non-sharded ttml::ops::cross_entropy_loss.
    ttnn::Tensor loss_val;
    if (reduce == ReduceType::MEAN) {
        auto s0 = ttnn::sum(per_pos, 0, /* keepdim */ true, std::nullopt, core::ComputeKernelConfig::precise());
        auto s2 = ttnn::sum(s0, 2, /* keepdim */ true, std::nullopt, core::ComputeKernelConfig::precise());
        loss_val = ttnn::multiply(s2, inv_N, ttnn::DataType::BFLOAT16);
    } else {
        loss_val = ttnn::typecast(per_pos, ttnn::DataType::BFLOAT16);
    }
    per_pos.deallocate();

    auto out = autograd::create_tensor(loss_val);

    // Backward: purely local, from the row state (M, log1p(T)); no communication and no statistics pass.
    //   dx_j = s * exp(x_j - M - l) on every shard, s = inv_N * upstream (folded into the kernel);
    //   dx_t = s * expm1((x_t - M) - l) on the target's shard, which is -s * T / (1 + T), the sum of all other
    //   gradients (Eq. 7), when the target is the selected maximum.
    autograd::GradFunction grad_fn = [logits, out, state, targets_raw, cluster_axis, inv_N]() {
        if (!out->is_grad_initialized()) {
            return;
        }
        logits->add_grad(
            metal::vocab_parallel_ce_bw(logits->get_value(), targets_raw, state, out->get_grad(), inv_N, cluster_axis));
    };

    out->set_node(autograd::add_backward_node(std::move(grad_fn), out, logits));
    return out;
}

}  // namespace ttml::ops::distributed
