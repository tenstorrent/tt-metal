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
#include "metal/ops/vp_ce_row_stats/vp_ce_row_stats.hpp"
#include "metal/ops/vp_cross_entropy_bw/device/vp_cross_entropy_bw_device_operation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

namespace {


}  // namespace

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
    const uint32_t local_V = logits_shape[3];
    const uint32_t N = B * S;
    // For MEAN we fold 1/N into (softmax−onehot) so the upstream scalar grad
    // multiplies through unchanged.  For NONE the upstream grad is the
    // per-position [B,1,S,1] tensor itself, so no extra factor is needed.
    const float inv_N = (reduce == ReduceType::MEAN) ? (1.0F / static_cast<float>(N)) : 1.0F;

    // Step 1: local max [B,1,S,1] per device: one pass over the shard (iteration 32: `vp_ce_row_stats`, per-slice
    // partials broadcast over the tile columns, max over them). fp32 holding bf16 values exactly.
    auto local_max = ttnn::max(ttml::metal::vp_ce_row_stats(logits->get_value(), 0U), 3, /* keepdim */ true);

    // Step 2: all-gather local maxes → [B,1,S,tp_size] → global max [B,1,S,1]
    auto all_max_val = ttnn_fixed::distributed::all_gather(local_max, 3, cluster_axis);
    auto global_max = ttnn::max(all_max_val, 3, /* keepdim */ true);

    // Step 3: the row sums of exp(logits - global_max) in one pass over the shard (iteration 32: `vp_ce_row_stats`
    // mode 1, fp32 exp on the SFPU, per-slice partial sums in column 0, summed); the 128 MB fp32 exp tensor of the
    // sub-exp + sum path is gone.
    auto local_sum = ttnn::sum(
        ttml::metal::vp_ce_row_stats(logits->get_value(), 1U, global_max),
        3,
        /* keepdim */ true,
        std::nullopt,
        core::ComputeKernelConfig::precise());
    auto global_sum = ttnn_fixed::distributed::all_reduce(local_sum, cluster_axis);

    // log_normalizer = global_max + log(global_sum)  [B,1,S,1]
    // global_max is FP32 (bf16 values); log(global_sum) is FP32.
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

    // Step 5: per-position loss = log_normalizer − target_logit  [B,1,S,1].
    // Always subtract in FP32. For MEAN we keep the FP32 tensor
    // for the precise sum reductions; for NONE we typecast to BF16 to match
    // the output dtype of the non-sharded ttml::ops::cross_entropy_loss.
    auto per_pos = ttnn::subtract(log_normalizer, target_logit, ttnn::DataType::FLOAT32);
    ttnn::Tensor loss_val;
    if (reduce == ReduceType::MEAN) {
        auto s0 = ttnn::sum(per_pos, 0, /* keepdim */ true, std::nullopt, core::ComputeKernelConfig::precise());
        auto s2 = ttnn::sum(s0, 2, /* keepdim */ true, std::nullopt, core::ComputeKernelConfig::precise());
        loss_val = ttnn::multiply(s2, inv_N, ttnn::DataType::BFLOAT16);
    } else {
        loss_val = ttnn::typecast(per_pos, ttnn::DataType::BFLOAT16);
    }

    auto out = autograd::create_tensor(loss_val);

    // ---------------------------------------------------------------
    // Backward  (purely local — no inter-device communication)
    //
    // dCE/dx_k = (softmax_k − onehot_k) * scale * grad_output
    //   scale = 1/N  for MEAN  → grad_output is scalar [1,1,1,1]
    //   scale = 1.0  for NONE  → grad_output is per-position [B,1,S,1]
    //
    // The same ttml_subtract_at_target kernel handles both: the `subtract_value`
    // it scatters at target positions matches the scale already applied to
    // softmax_k, so the resulting tile encodes (softmax_k − onehot_k) * scale
    // before the upstream-grad broadcast multiply.  The trailing
    // ttnn::multiply broadcasts the scalar grad over [B,1,S,V/tp] for MEAN
    // and broadcasts the per-position grad over the local vocab dim for NONE
    // — the same broadcast pattern used by the forward-pass max/exp path.
    // ---------------------------------------------------------------
    // One pass over the local shard (iteration 14 of the Mistral loop): grad = inv_N * g * (exp(x - logZ) - onehot),
    // in place of subtract+exp (fp32 intermediate), multiply, subtract_at_target and a multiply by g.
    autograd::GradFunction grad_fn = [logits, out, log_normalizer, targets_raw, local_V, cluster_axis, inv_N]() {
        if (!out->is_grad_initialized()) {
            return;
        }
        // The kernel multiplies by a full tile of g (its DEST operand is a tile): repeat the [1, 1, 1, 1] gradient.
        auto g_tile =
            ttnn::repeat(out->get_grad(), ttnn::Shape{1U, 1U, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH});
        logits->add_grad(ttnn::prim::ttml_vp_cross_entropy_bw(
            logits->get_value(),
            log_normalizer,
            targets_raw,
            g_tile,
            /*local_V=*/local_V,
            /*cluster_axis=*/cluster_axis,
            /*first_v=*/0U,
            /*scale=*/inv_N));
    };

    out->set_node(autograd::add_backward_node(std::move(grad_fn), out, logits));
    return out;
}

}  // namespace ttml::ops::distributed
