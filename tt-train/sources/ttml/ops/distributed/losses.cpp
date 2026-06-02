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
#include "metal/ops/select_target_logit/device/select_target_logit_device_operation.hpp"
#include "metal/ops/subtract_at_target/device/subtract_at_target_device_operation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

namespace {

// Fuses `exp` after `subtract` into a single binary_ng kernel via the SFPU
// op-chain (post_activations).  The (a - b) intermediate stays in DST regs.
ttnn::Tensor fused_subtract_exp_fp32(const ttnn::Tensor& a, const ttnn::Tensor& b) {
    using EltwiseUnary = ttnn::operations::unary::EltwiseUnaryWithParam;
    // 0.0f matches ttnn::exp's default fast_and_approximate_mode=false.
    const EltwiseUnary exp_act{ttnn::operations::unary::UnaryOpType::EXP, 0.0f};
    const ttsl::Span<const EltwiseUnary> post_activations(&exp_act, 1);
    return ttnn::subtract(
        a,
        b,
        /*output_dtype=*/ttnn::DataType::FLOAT32,
        /*memory_config=*/std::nullopt,
        /*output=*/std::nullopt,
        post_activations);
}

}  // namespace

autograd::TensorPtr vocab_parallel_cross_entropy_loss(
    const autograd::TensorPtr& logits, const autograd::TensorPtr& targets, std::optional<uint32_t> cluster_axis) {
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
    const uint32_t local_V = logits_shape[3];
    const uint32_t N = B * S;

    // Step 1: local max [B,1,S,1] per device (BF16 — no precision loss)
    auto local_max = ttnn::max(logits->get_value(), 3, /* keepdim */ true);

    // Step 2: all-gather local maxes → [B,1,S,tp_size] → global max [B,1,S,1]
    auto all_max_val = ttnn_fixed::distributed::all_gather(local_max, 3, cluster_axis);
    auto global_max = ttnn::max(all_max_val, 3, /* keepdim */ true);

    // Step 3: fused (logits - global_max).exp() into FP32 — single binary_ng kernel,
    // no intermediate [B,1,S,V/tp_size] FP32 tensor.
    auto local_exp = fused_subtract_exp_fp32(logits->get_value(), global_max);
    auto local_sum = ttnn::sum(local_exp, 3, /* keepdim */ true, std::nullopt, core::ComputeKernelConfig::precise());
    auto global_sum = ttnn_fixed::distributed::all_reduce(local_sum, cluster_axis);

    // log_normalizer = global_max + log(global_sum)  [B,1,S,1]
    // global_max is BF16; log(global_sum) is FP32 — output_dtype promotes the add.
    auto log_normalizer = ttnn::add(global_max, ttnn::log(global_sum), ttnn::DataType::FLOAT32);

    // Step 4: target_logit_local via select_target_logit:
    // Each TP device k owns vocab [k*local_V, (k+1)*local_V).
    auto targets_raw = targets->get_value();
    if (targets_raw.layout() != ttnn::Layout::ROW_MAJOR) {
        targets_raw = ttnn::to_layout(targets_raw, ttnn::Layout::ROW_MAJOR);
    }
    if (targets_raw.logical_shape().rank() != 2U) {
        targets_raw = ttnn::reshape(targets_raw, ttnn::Shape({B, S}));
    }
    const auto& logits_val = logits->get_value();
    auto gather_output = ttnn::prim::ttml_select_target_logit(
        logits_val, targets_raw, /*local_V=*/local_V, /*cluster_axis=*/cluster_axis, /*first_v=*/0U);

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
    // Note we scale by 1/N first, cast to BF16, then scatter-subtract 1/N at the
    // target position; the ordering matches the reference cross_entropy_bw kernel,
    // whose writer does the onehot subtraction directly on the already-scaled BF16 tile.
    // ---------------------------------------------------------------
    autograd::GradFunction grad_fn = [logits, out, global_max, global_sum, targets_raw, local_V, cluster_axis, N]() {
        if (!out->is_grad_initialized()) {
            return;
        }
        const float inv_N = 1.0F / static_cast<float>(N);

        auto local_exp = fused_subtract_exp_fp32(logits->get_value(), global_max);

        auto softmax_k = ttnn::multiply(local_exp, ttnn::reciprocal(global_sum));
        auto scaled_softmax = ttnn::multiply(softmax_k, inv_N, ttnn::DataType::BFLOAT16);

        // Single mesh workload — the program factory derives each device's shard window
        // from its mesh coordinate, mirroring the forward-pass select_target_logit call.
        ttnn::prim::ttml_subtract_at_target(
            scaled_softmax,
            targets_raw,
            /*local_V=*/local_V,
            /*cluster_axis=*/cluster_axis,
            /*first_v=*/0U,
            scaled_softmax,
            /*subtract_value=*/inv_N);

        logits->add_grad(ttnn::multiply(scaled_softmax, out->get_grad(), ttnn::DataType::BFLOAT16));
    };

    out->set_node(autograd::add_backward_node(std::move(grad_fn), out, logits));
    return out;
}

}  // namespace ttml::ops::distributed
