// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_sdpa.hpp"

#include <fmt/core.h>

#include <cmath>
#include <limits>
#include <tt-umd-workload/cluster.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/ops/ring_sdpa_bw/ring_sdpa_bw.hpp"
#include "metal/ops/ring_sdpa_fw/ring_sdpa_fw.hpp"
#include "ops/binary_ops.hpp"
#include "ops/distributed/comm_ops.hpp"
#include "ops/distributed/ring_attention_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

namespace {

constexpr auto kZigZagFwDirection = ttml::metal::ops::ring_sdpa_fw::RingDirection::Backward;
constexpr auto kZigZagBwDirection = ttml::metal::ops::ring_sdpa_bw::RingDirection::Backward;

struct RingSpec {
    uint32_t ring_size = 0U;
    uint32_t cp_axis = 0U;
};

// FP32 running output and logsumexp of each sequence half.
struct ZigZagForwardAccum {
    SeqHalves output;
    SeqHalves lse;
};

struct ZigZagForwardBuffers {
    ttnn::Tensor no_contrib_intermediates;
    ttnn::Tensor cross_output;
    ttnn::Tensor cross_intermediates;
    std::array<ttnn::Tensor, 2> chunk_output;
    std::array<ttnn::Tensor, 2> chunk_intermediates;
};

ZigZagForwardAccum zigzag_forward_step0(
    const ttnn::Tensor& query, const ttnn::Tensor& key, const ttnn::Tensor& value, const RingSpec& ring) {
    const uint32_t seq_len = query.logical_shape()[2];
    const uint32_t half = seq_len / 2U;
    ttnn::Tensor output = ttnn::empty_like(query);
    ttnn::Tensor intermediates = make_intermediates(query, seq_len);
    ttml::metal::ring_sdpa_fw(
        {{.query = query, .key = key, .value = value, .output = output, .intermediates = intermediates}},
        ring.ring_size,
        ring.cp_axis,
        /*step=*/0U,
        ttml::metal::AttentionMaskType::Causal,
        kZigZagFwDirection);
    return {
        .output = split_seq(ttnn::typecast(output, ttnn::DataType::FLOAT32)),
        .lse = {
            slice_seq(intermediates, 0U, half, /*width=*/1U), slice_seq(intermediates, half, seq_len, /*width=*/1U)}};
}

ZigZagForwardBuffers make_zigzag_forward_buffers(const ttnn::Tensor& query, const ttnn::Tensor& q_half) {
    const uint32_t half = q_half.logical_shape()[2];
    // Chunk outputs must be finite where a rank does not write them: the merge multiplies them by exp(-inf) = 0.
    ZigZagForwardBuffers buffers{
        .no_contrib_intermediates = make_no_contrib_intermediates(query, half),
        .cross_output = ttnn::empty_like(q_half),
        .cross_intermediates = make_intermediates(query, half),
        .chunk_output = {ttnn::zeros_like(q_half), ttnn::zeros_like(q_half)},
        .chunk_intermediates = {make_intermediates(query, half), make_intermediates(query, half)}};
    ttnn::copy(buffers.no_contrib_intermediates, buffers.chunk_intermediates[0]);
    return buffers;
}

void zigzag_forward_step(
    const SeqHalves& q,
    const SeqHalves& k,
    const SeqHalves& v,
    uint32_t step,
    const RingSpec& ring,
    ZigZagForwardBuffers& buffers,
    ZigZagForwardAccum& accum) {
    using ttml::metal::AttentionMaskType;
    ttml::metal::ring_sdpa_fw(
        {{.query = q.second,
          .key = k.first,
          .value = v.first,
          .output = buffers.cross_output,
          .intermediates = buffers.cross_intermediates}},
        ring.ring_size,
        ring.cp_axis,
        step,
        AttentionMaskType::None,
        kZigZagFwDirection);

    // A rank computes chunk 1 for its first steps, then chunk 0. Reset chunk 1's lse to -inf every step so the
    // merge ignores it after the switch; chunk 0 starts at -inf and is written at every step after the switch.
    ttnn::copy(buffers.no_contrib_intermediates, buffers.chunk_intermediates[1]);
    ttml::metal::ring_sdpa_fw(
        {{.query = q.first,
          .key = k.first,
          .value = v.first,
          .output = buffers.chunk_output[0],
          .intermediates = buffers.chunk_intermediates[0]},
         {.query = q.second,
          .key = k.second,
          .value = v.second,
          .output = buffers.chunk_output[1],
          .intermediates = buffers.chunk_intermediates[1]}},
        ring.ring_size,
        ring.cp_axis,
        step,
        AttentionMaskType::Causal,
        kZigZagFwDirection,
        ttml::metal::ops::RingLoadBalancing::ZigZag);

    online_softmax_merge(accum.output.second, accum.lse.second, buffers.cross_output, buffers.cross_intermediates);
    online_softmax_merge(accum.output.first, accum.lse.first, buffers.chunk_output[0], buffers.chunk_intermediates[0]);
    online_softmax_merge(
        accum.output.second, accum.lse.second, buffers.chunk_output[1], buffers.chunk_intermediates[1]);
}

// What the zig-zag backward keeps from the forward.
struct ZigZagSaved {
    autograd::TensorPtr query;
    autograd::TensorPtr key;
    autograd::TensorPtr value;
    autograd::TensorPtr out;
    SeqHalves lse;
    SeqHalves k_current;
    SeqHalves v_current;
    std::vector<SeqHalves> k_steps;
    std::vector<SeqHalves> v_steps;
    RingSpec ring;
};

struct ZigZagBackwardInputs {
    SeqHalves query;
    SeqHalves attn_output;
    SeqHalves grad_output;
    SeqHalves intermediates;
};

struct ZigZagGradBuffers {
    ttnn::Tensor zero_Q_half;
    ttnn::Tensor zero_K_half;
    ttnn::Tensor zero_V_half;
    ttnn::Tensor grad_Q_cross;
    ttnn::Tensor grad_K_cross;
    ttnn::Tensor grad_V_cross;
    std::array<ttnn::Tensor, 2> grad_Q_chunk;
    std::array<ttnn::Tensor, 2> grad_K_chunk;
    std::array<ttnn::Tensor, 2> grad_V_chunk;
};

// FP32 running grads of each sequence half.
struct ZigZagGradAccum {
    SeqHalves query;
    SeqHalves key;
    SeqHalves value;
};

ZigZagGradBuffers make_zigzag_grad_buffers(
    const ttnn::Tensor& q_half, const ttnn::Tensor& key, const ttnn::Tensor& value) {
    const uint32_t half = q_half.logical_shape()[2];
    const ttnn::Tensor zero_Q = ttnn::zeros_like(q_half);
    const ttnn::Tensor zero_K = ttnn::zeros_like(slice_seq(key, 0U, half));
    const ttnn::Tensor zero_V = ttnn::zeros_like(slice_seq(value, 0U, half));
    return {
        .zero_Q_half = zero_Q,
        .zero_K_half = zero_K,
        .zero_V_half = zero_V,
        .grad_Q_cross = ttnn::empty_like(zero_Q),
        .grad_K_cross = ttnn::empty_like(zero_K),
        .grad_V_cross = ttnn::empty_like(zero_V),
        .grad_Q_chunk = {ttnn::zeros_like(zero_Q), ttnn::zeros_like(zero_Q)},
        .grad_K_chunk = {ttnn::zeros_like(zero_K), ttnn::zeros_like(zero_K)},
        .grad_V_chunk = {ttnn::zeros_like(zero_V), ttnn::zeros_like(zero_V)}};
}

ZigZagGradAccum make_zigzag_grad_accum(const ZigZagGradBuffers& buffers) {
    return {
        .query =
            {ttnn::zeros_like(buffers.zero_Q_half, ttnn::DataType::FLOAT32),
             ttnn::zeros_like(buffers.zero_Q_half, ttnn::DataType::FLOAT32)},
        .key =
            {ttnn::zeros_like(buffers.zero_K_half, ttnn::DataType::FLOAT32),
             ttnn::zeros_like(buffers.zero_K_half, ttnn::DataType::FLOAT32)},
        .value = {
            ttnn::zeros_like(buffers.zero_V_half, ttnn::DataType::FLOAT32),
            ttnn::zeros_like(buffers.zero_V_half, ttnn::DataType::FLOAT32)}};
}

void zigzag_backward_step(
    const ZigZagBackwardInputs& in,
    const SeqHalves& k,
    const SeqHalves& v,
    uint32_t step,
    const RingSpec& ring,
    ZigZagGradBuffers& buffers) {
    using ttml::metal::AttentionMaskType;
    ttml::metal::ring_sdpa_bw(
        {{.grad_output = in.grad_output.second,
          .attn_output = in.attn_output.second,
          .query = in.query.second,
          .key = k.first,
          .value = v.first,
          .intermediates = in.intermediates.second,
          .grad_query = buffers.grad_Q_cross,
          .grad_key = buffers.grad_K_cross,
          .grad_value = buffers.grad_V_cross}},
        ring.ring_size,
        ring.cp_axis,
        step,
        AttentionMaskType::None,
        kZigZagBwDirection);

    // Steps run backwards here, so a rank computes chunk 0 first, then chunk 1. Both chunks' grads are summed,
    // so zero chunk 0 every step: after the switch it would add its last grads again.
    ttnn::copy(buffers.zero_Q_half, buffers.grad_Q_chunk[0]);
    ttnn::copy(buffers.zero_K_half, buffers.grad_K_chunk[0]);
    ttnn::copy(buffers.zero_V_half, buffers.grad_V_chunk[0]);
    ttml::metal::ring_sdpa_bw(
        {{.grad_output = in.grad_output.first,
          .attn_output = in.attn_output.first,
          .query = in.query.first,
          .key = k.first,
          .value = v.first,
          .intermediates = in.intermediates.first,
          .grad_query = buffers.grad_Q_chunk[0],
          .grad_key = buffers.grad_K_chunk[0],
          .grad_value = buffers.grad_V_chunk[0]},
         {.grad_output = in.grad_output.second,
          .attn_output = in.attn_output.second,
          .query = in.query.second,
          .key = k.second,
          .value = v.second,
          .intermediates = in.intermediates.second,
          .grad_query = buffers.grad_Q_chunk[1],
          .grad_key = buffers.grad_K_chunk[1],
          .grad_value = buffers.grad_V_chunk[1]}},
        ring.ring_size,
        ring.cp_axis,
        step,
        AttentionMaskType::Causal,
        kZigZagBwDirection,
        ttml::metal::ops::RingLoadBalancing::ZigZag);
}

// The cross block's grads go to Q's second half and K/V's first half; chunk c's grads go to half c of Q, K and V.
void accumulate_zigzag_grads(const ZigZagGradBuffers& buffers, ZigZagGradAccum& accum) {
    constexpr auto kFp32 = ttnn::DataType::FLOAT32;
    accum.query.first = ttnn::add(accum.query.first, ttnn::typecast(buffers.grad_Q_chunk[0], kFp32));
    accum.query.second = ttnn::add(
        accum.query.second,
        ttnn::add(ttnn::typecast(buffers.grad_Q_cross, kFp32), ttnn::typecast(buffers.grad_Q_chunk[1], kFp32)));
    accum.key.first = ttnn::add(
        accum.key.first,
        ttnn::add(ttnn::typecast(buffers.grad_K_cross, kFp32), ttnn::typecast(buffers.grad_K_chunk[0], kFp32)));
    accum.key.second = ttnn::add(accum.key.second, ttnn::typecast(buffers.grad_K_chunk[1], kFp32));
    accum.value.first = ttnn::add(
        accum.value.first,
        ttnn::add(ttnn::typecast(buffers.grad_V_cross, kFp32), ttnn::typecast(buffers.grad_V_chunk[0], kFp32)));
    accum.value.second = ttnn::add(accum.value.second, ttnn::typecast(buffers.grad_V_chunk[1], kFp32));
}

// Step 0 is one causal call over the full local sequence; its grads join the ring steps' and go to the inputs.
void finish_zigzag_backward(
    const ZigZagSaved& saved, const ZigZagGradAccum& accum, const ttnn::Tensor& global_intermediates) {
    constexpr auto kFp32 = ttnn::DataType::FLOAT32;
    const ttnn::Tensor& query = saved.query->get_value();
    const ttnn::Tensor& key = saved.key->get_value();
    const ttnn::Tensor& value = saved.value->get_value();
    const ttnn::Tensor grad_Q_step0 = ttnn::empty_like(query);
    const ttnn::Tensor grad_K_step0 = ttnn::empty_like(key);
    const ttnn::Tensor grad_V_step0 = ttnn::empty_like(value);
    ttml::metal::ring_sdpa_bw(
        {{.grad_output = saved.out->get_grad(),
          .attn_output = saved.out->get_value(),
          .query = query,
          .key = key,
          .value = value,
          .intermediates = global_intermediates,
          .grad_query = grad_Q_step0,
          .grad_key = grad_K_step0,
          .grad_value = grad_V_step0}},
        saved.ring.ring_size,
        saved.ring.cp_axis,
        /*step=*/0U,
        ttml::metal::AttentionMaskType::Causal,
        kZigZagBwDirection);
    const ttnn::Tensor grad_Q = ttnn::add(
        ttnn::concat(std::vector<ttnn::Tensor>{accum.query.first, accum.query.second}, /*dim=*/2),
        ttnn::typecast(grad_Q_step0, kFp32));
    const ttnn::Tensor grad_K = ttnn::add(
        ttnn::concat(std::vector<ttnn::Tensor>{accum.key.first, accum.key.second}, /*dim=*/2),
        ttnn::typecast(grad_K_step0, kFp32));
    const ttnn::Tensor grad_V = ttnn::add(
        ttnn::concat(std::vector<ttnn::Tensor>{accum.value.first, accum.value.second}, /*dim=*/2),
        ttnn::typecast(grad_V_step0, kFp32));

    saved.query->add_grad(ttnn::typecast(grad_Q, query.dtype()));
    saved.key->add_grad(ttnn::typecast(grad_K, key.dtype()));
    saved.value->add_grad(ttnn::typecast(grad_V, value.dtype()));
}

void zigzag_backward(ZigZagSaved& saved) {
    using ttnn_fixed::distributed::RingShiftDirection;
    const ttnn::Tensor global_intermediates = pad_lse_to_intermediates_layout(
        ttnn::concat(std::vector<ttnn::Tensor>{saved.lse.first, saved.lse.second}, /*dim=*/2));
    const ZigZagBackwardInputs in{
        .query = split_seq(saved.query->get_value()),
        .attn_output = split_seq(saved.out->get_value()),
        .grad_output = split_seq(saved.out->get_grad()),
        .intermediates = split_seq(global_intermediates)};
    ZigZagGradBuffers buffers =
        make_zigzag_grad_buffers(in.query.first, saved.key->get_value(), saved.value->get_value());
    ZigZagGradAccum accum = make_zigzag_grad_accum(buffers);

    const uint32_t cp_axis = saved.ring.cp_axis;
    for (uint32_t step = saved.ring.ring_size - 1U; step >= 1U; --step) {
        const SeqHalves& k = saved.k_steps.empty() ? saved.k_current : saved.k_steps[step];
        const SeqHalves& v = saved.v_steps.empty() ? saved.v_current : saved.v_steps[step];
        zigzag_backward_step(in, k, v, step, saved.ring, buffers);
        accumulate_zigzag_grads(buffers, accum);

        if (saved.k_steps.empty() && step > 1U) {
            saved.k_current = ring_shift_halves(saved.k_current, cp_axis, RingShiftDirection::Forward);
            saved.v_current = ring_shift_halves(saved.v_current, cp_axis, RingShiftDirection::Forward);
        }
        accum.key = ring_shift_halves(accum.key, cp_axis, RingShiftDirection::Forward);
        accum.value = ring_shift_halves(accum.value, cp_axis, RingShiftDirection::Forward);
    }
    finish_zigzag_backward(saved, accum, global_intermediates);
}

autograd::TensorPtr ring_attention_sdpa_zigzag(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const uint32_t ring_size,
    const uint32_t cp_axis_value,
    const bool reuse_forward_kv) {
    using ttnn_fixed::distributed::RingShiftDirection;
    const auto& query_tensor = query->get_value();
    const uint32_t seq_len_local = query_tensor.logical_shape()[2];
    TT_FATAL(
        seq_len_local % (2U * tt::constants::TILE_HEIGHT) == 0U,
        "Zig-zag ring attention splits the local sequence ({} tokens) into two halves, so it must be a multiple of "
        "{}",
        seq_len_local,
        2U * tt::constants::TILE_HEIGHT);
    const RingSpec ring{.ring_size = ring_size, .cp_axis = cp_axis_value};

    const SeqHalves q_halves = split_seq(query_tensor);
    ZigZagForwardAccum accum = zigzag_forward_step0(query_tensor, key->get_value(), value->get_value(), ring);
    ZigZagForwardBuffers buffers = make_zigzag_forward_buffers(query_tensor, q_halves.first);

    std::vector<SeqHalves> k_steps(reuse_forward_kv ? ring_size : 0U);
    std::vector<SeqHalves> v_steps(reuse_forward_kv ? ring_size : 0U);
    SeqHalves k_current = split_seq(key->get_value());
    SeqHalves v_current = split_seq(value->get_value());
    for (uint32_t step = 1U; step < ring_size; ++step) {
        k_current = ring_shift_halves(k_current, cp_axis_value, RingShiftDirection::Backward);
        v_current = ring_shift_halves(v_current, cp_axis_value, RingShiftDirection::Backward);
        if (reuse_forward_kv) {
            k_steps[step] = k_current;
            v_steps[step] = v_current;
        }
        zigzag_forward_step(q_halves, k_current, v_current, step, ring, buffers, accum);
    }

    auto out = autograd::create_tensor(ttnn::typecast(
        ttnn::concat(std::vector<ttnn::Tensor>{accum.output.first, accum.output.second}, /*dim=*/2),
        query_tensor.dtype()));
    if (reuse_forward_kv) {
        // The backward reads k_steps / v_steps; do not hold a second reference to the last step's K/V.
        k_current = SeqHalves{ttnn::Tensor(), ttnn::Tensor()};
        v_current = SeqHalves{ttnn::Tensor(), ttnn::Tensor()};
    }

    autograd::GradFunction grad_fn = [saved = ZigZagSaved{
                                          .query = query,
                                          .key = key,
                                          .value = value,
                                          .out = out,
                                          .lse = std::move(accum.lse),
                                          .k_current = std::move(k_current),
                                          .v_current = std::move(v_current),
                                          .k_steps = std::move(k_steps),
                                          .v_steps = std::move(v_steps),
                                          .ring = ring}]() mutable { zigzag_backward(saved); };
    out->set_node(autograd::add_backward_node(std::move(grad_fn), out, query, key, value));
    return out;
}

}  // namespace

autograd::TensorPtr ring_attention_sdpa(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const std::optional<autograd::TensorPtr>& mask,
    const ttml::metal::AttentionMaskType mask_type,
    const std::optional<bool> reuse_forward_kv_requested,
    const std::optional<core::distributed::CpSharding> cp_sharding) {
    if (!autograd::ctx().is_parallelism_context_initialized() ||
        !autograd::ctx().get_parallelism_context().is_cp_enabled()) {
        return ttml::ops::scaled_dot_product_attention(query, key, value, mask);
    }

    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis_value = pctx.get_cp_axis().value();
    const uint32_t ring_size = pctx.get_cp_size();
    const bool reuse_forward_kv = reuse_forward_kv_requested.value_or(pctx.get_cp_config().reuse_forward_kv) &&
                                  autograd::ctx().get_gradient_mode() == autograd::GradMode::ENABLED &&
                                  autograd::any_requires_grad(query, key, value);
    const auto& query_tensor = query->get_value();
    auto* mesh_device = query_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Query tensor must be on a mesh device for ring attention");
    TT_FATAL(
        !mask.has_value(),
        "Non-causal mask is not supported in CP mode for now, pass nullopt if you want to use causal mask");
    TT_FATAL(
        mask_type != ttml::metal::AttentionMaskType::Arbitrary,
        "Arbitrary attention mask is not supported in CP mode, use None or Causal");

    switch (cp_sharding.value_or(pctx.get_cp_config().sharding)) {
        case core::distributed::CpSharding::Contiguous: break;
        case core::distributed::CpSharding::ZigZag:
            TT_FATAL(
                mask_type == ttml::metal::AttentionMaskType::Causal,
                "Zig-zag CP sharding balances causal attention; use contiguous CP sharding without a causal mask");
            return ring_attention_sdpa_zigzag(query, key, value, ring_size, cp_axis_value, reuse_forward_kv);
    }

    auto [batch_num, heads, seq_len_local, dim] = query_tensor.logical_shape().to_array_4D();
    // Initialize current K and V (will be ring-shifted each step)
    // Use raw ttnn::Tensor since we define backward manually
    ttnn::Tensor k_current = key->get_value();
    ttnn::Tensor v_current = value->get_value();

    // Backward at ring step s needs exactly the K/V the forward used at step s. Keeping those handles saves
    // backward its 2 * (ring_size - 1) K/V shifts, at the cost of 2 * (ring_size - 2) extra K-sized tensors
    // per device staying alive until this node's backward has run, or until this op returns when no backward
    // node is registered.
    // k_steps[s] = K used at ring step s, for s < ring_size - 1; the last step's K/V stay in k_current/v_current
    std::vector<ttnn::Tensor> k_steps(reuse_forward_kv ? ring_size - 1U : 0U);
    std::vector<ttnn::Tensor> v_steps(reuse_forward_kv ? ring_size - 1U : 0U);

    // Initialize accumulators for online softmax.
    // output_accum: weighted sum of outputs from all steps. Kept FP32: the online-softmax
    // rescaling rewrites the whole accumulator every ring step, so a bf16 accumulator
    // compounds ~ring_size rounding errors into the saved O. Backward consumes that O in
    // u = rowsum(dO * O), where the error is amplified by the softmax-backward
    // cancellation in (dP - u) — enough to push dK outside test tolerance.
    ttnn::Tensor output_accum = ttnn::full(
        ttnn::Shape{batch_num, heads, seq_len_local, dim},
        0.0F,
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        std::ref(*mesh_device));

    // global_lse: running logsumexp across all steps
    // lse = log(sum(exp(scale * score_i))) — the log of the softmax normalizer
    // Initialized to -inf (no contribution: exp(-inf) = 0)
    ttnn::Tensor global_lse = ttnn::full(
        ttnn::Shape{batch_num, heads, seq_len_local, 1U},
        -std::numeric_limits<float>::infinity(),
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        std::ref(*mesh_device));

    ttnn::Tensor output_tensor = ttnn::empty_like(query_tensor);
    ttnn::Tensor intermediate_tensor = make_intermediates(query_tensor, seq_len_local);
    ttnn::Tensor no_contrib_intermediate = make_no_contrib_intermediates(query_tensor, seq_len_local);

    for (uint32_t step = 0; step < ring_size; ++step) {
        // For causal masking, initialize intermediate_tensor to "no contribution" values
        // Devices that are skipped will keep these values, indicating zero contribution
        if (mask_type == ttml::metal::AttentionMaskType::Causal) {
            ttnn::copy(no_contrib_intermediate, intermediate_tensor);
        }

        ttml::metal::ring_sdpa_fw(
            {{.query = query_tensor,
              .key = k_current,
              .value = v_current,
              .output = output_tensor,
              .intermediates = intermediate_tensor}},
            ring_size,
            cp_axis_value,
            step,
            mask_type,
            ttml::metal::ops::ring_sdpa_fw::RingDirection::Backward);

        // Fold this step's partial result into the FP32 accumulator and the running lse. Step 0 writes every row of
        // output_tensor on every device, so a step a device skips re-merges a finite output with weight 0.
        online_softmax_merge(output_accum, global_lse, output_tensor, intermediate_tensor);

        if (step < ring_size - 1) {
            if (reuse_forward_kv) {
                k_steps[step] = k_current;
                v_steps[step] = v_current;
            }
            k_current = ttnn_fixed::distributed::ring_shift(
                k_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Backward);
            v_current = ttnn_fixed::distributed::ring_shift(
                v_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Backward);
        }
    }

    // Single rounding to the input dtype; the graph (and the saved O used by backward)
    // sees the same dtype as before.
    auto out = autograd::create_tensor(ttnn::typecast(output_accum, query_tensor.dtype()));
    ttnn::Tensor final_lse = global_lse;

    autograd::GradFunction grad_fn = [query,
                                      key,
                                      value,
                                      out,
                                      final_lse,
                                      k_current,  // K at end of forward (position k1)
                                      v_current,  // V at end of forward (position v1)
                                      k_steps,
                                      v_steps,
                                      ring_size,
                                      cp_axis_value,
                                      mask_type]() mutable {
        const auto& grad_output = out->get_grad();
        const auto& attn_output = out->get_value();
        const auto& query_tensor = query->get_value();

        // FP32 host accumulators: each ring step contributes a bf16 kernel output, but
        // summing them in bf16 rounds the full running magnitude every step. The
        // accumulators are cast back to the input dtype once, after the loop.
        ttnn::Tensor grad_Q_accum = ttnn::zeros_like(query_tensor, ttnn::DataType::FLOAT32);
        ttnn::Tensor grad_K_accum = ttnn::zeros_like(key->get_value(), ttnn::DataType::FLOAT32);
        ttnn::Tensor grad_V_accum = ttnn::zeros_like(value->get_value(), ttnn::DataType::FLOAT32);

        ttnn::Tensor grad_Q_step = ttnn::zeros_like(query_tensor);
        ttnn::Tensor grad_K_step = ttnn::zeros_like(key->get_value());
        ttnn::Tensor grad_V_step = ttnn::zeros_like(value->get_value());

        // Standard ring-flash-attention backward: every step gets the UNSCALED upstream
        // gradient, the GLOBAL forward output, and the GLOBAL logsumexp. The sdpa_bw
        // kernels recompute P = exp(scale*QK^T - lse) from the supplied lse, so the global
        // lse yields the global attention weights restricted to this chunk's columns, and
        // u = rowsum(dO * O_global) is the global softmax-backward correction. Per-chunk
        // contributions then sum to the exact dQ/dK/dV. Feeding per-chunk O/lse here would
        // bias dQ/dK (per-chunk u instead of global) — only dV would come out right.
        ttnn::Tensor global_intermediates = pad_lse_to_intermediates_layout(final_lse);

        // Zero sources for resetting the step buffers before every ring step.
        const ttnn::Tensor zero_Q = ttnn::zeros_like(grad_Q_step);
        const ttnn::Tensor zero_K = ttnn::zeros_like(grad_K_step);
        const ttnn::Tensor zero_V = ttnn::zeros_like(grad_V_step);

        // Loop over ring steps in reverse order (from last to first)
        for (int step = ring_size - 1; step >= 0; --step) {
            const uint32_t step_idx = step;

            // Devices skipped by the causal schedule at this step do not run the kernels,
            // so their step buffers must be zeroed to contribute nothing to the accumulators.
            ttnn::copy(zero_Q, grad_Q_step);
            ttnn::copy(zero_K, grad_K_step);
            ttnn::copy(zero_V, grad_V_step);

            // Use Backward direction (same as forward) since src = (device + step) % ring_size
            ttml::metal::ring_sdpa_bw(
                {{.grad_output = grad_output,
                  .attn_output = attn_output,  // Global forward output saved by autograd
                  .query = query_tensor,
                  .key = k_current,                       // K at current ring position
                  .value = v_current,                     // V at current ring position
                  .intermediates = global_intermediates,  // Global logsumexp in intermediates layout
                  .grad_query = grad_Q_step,
                  .grad_key = grad_K_step,
                  .grad_value = grad_V_step}},
                ring_size,
                cp_axis_value,
                step_idx,
                mask_type,
                ttml::metal::ops::ring_sdpa_bw::RingDirection::Backward);

            // Upcast so the accumulation stays FP32.
            grad_Q_accum = ttnn::add(grad_Q_accum, ttnn::typecast(grad_Q_step, ttnn::DataType::FLOAT32));
            grad_K_accum = ttnn::add(grad_K_accum, ttnn::typecast(grad_K_step, ttnn::DataType::FLOAT32));
            grad_V_accum = ttnn::add(grad_V_accum, ttnn::typecast(grad_V_step, ttnn::DataType::FLOAT32));

            // Move to the previous ring step.
            // K/V: the forward's kept handles, or shifted in FORWARD direction when none were kept
            // grad_K/V: always shifted in FORWARD direction, routes accumulated gradients back to correct device
            if (step > 0) {
                if (k_steps.empty()) {
                    k_current = ttnn_fixed::distributed::ring_shift(
                        k_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
                    v_current = ttnn_fixed::distributed::ring_shift(
                        v_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
                } else {
                    k_current = k_steps[step - 1];
                    v_current = v_steps[step - 1];
                }

                // Shift grad accumulators
                grad_K_accum = ttnn_fixed::distributed::ring_shift(
                    grad_K_accum, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
                grad_V_accum = ttnn_fixed::distributed::ring_shift(
                    grad_V_accum, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
            }
        }

        // Apply gradients, rounded once to the parameter dtype
        query->add_grad(ttnn::typecast(grad_Q_accum, query_tensor.dtype()));
        key->add_grad(ttnn::typecast(grad_K_accum, key->get_value().dtype()));
        value->add_grad(ttnn::typecast(grad_V_accum, value->get_value().dtype()));
    };

    out->set_node(autograd::add_backward_node(std::move(grad_fn), out, query, key, value));

    return out;
}

}  // namespace ttml::ops::distributed
