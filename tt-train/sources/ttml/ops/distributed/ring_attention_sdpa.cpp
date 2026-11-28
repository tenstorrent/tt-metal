// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_sdpa.hpp"

#include <fmt/core.h>

#include <cmath>
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
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

namespace {

using ttml::metal::AttentionMaskType;

constexpr auto kRingFwDirection = ttml::metal::ops::ring_sdpa_fw::RingDirection::Backward;
constexpr auto kRingBwDirection = ttml::metal::ops::ring_sdpa_bw::RingDirection::Backward;
constexpr auto kFp32 = ttnn::DataType::FLOAT32;

struct RingSpec {
    uint32_t ring_size = 0U;
    uint32_t cp_axis = 0U;
};

struct RingTensors {
    autograd::TensorPtr query;
    autograd::TensorPtr key;
    autograd::TensorPtr value;
    autograd::TensorPtr out;
};

// FP32 running output and logsumexp.
struct ForwardAccum {
    ttnn::Tensor output;
    ttnn::Tensor lse;
};

struct RingGrads {
    ttnn::Tensor query;
    ttnn::Tensor key;
    ttnn::Tensor value;
};

// Step 0 attends to the device's own K/V, so every device writes every row of output and intermediates.
// The running output is FP32: the online-softmax rescaling rewrites the whole accumulator every ring step, so a
// bf16 accumulator compounds ~ring_size rounding errors into the saved O. Backward consumes that O in
// u = rowsum(dO * O), where the softmax-backward cancellation in (dP - u) amplifies the error enough to push dK
// outside test tolerance.
ForwardAccum forward_step0(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const RingSpec& ring,
    AttentionMaskType mask_type,
    const ttnn::Tensor& output,
    const ttnn::Tensor& intermediates) {
    ttml::metal::ring_sdpa_fw(
        {{.query = query, .key = key, .value = value, .output = output, .intermediates = intermediates}},
        ring.ring_size,
        ring.cp_axis,
        /*step=*/0U,
        mask_type,
        kRingFwDirection);
    return {
        .output = ttnn::typecast(output, kFp32),
        .lse = slice_seq(intermediates, 0U, query.logical_shape()[2], /*width=*/1U)};
}

// Step 0 runs over the full local sequence, so it writes every grad row on every device.
RingGrads ring_backward_step0(
    const RingTensors& tensors,
    const ttnn::Tensor& global_intermediates,
    const RingSpec& ring,
    AttentionMaskType mask_type) {
    const ttnn::Tensor& query = tensors.query->get_value();
    const ttnn::Tensor& key = tensors.key->get_value();
    const ttnn::Tensor& value = tensors.value->get_value();
    RingGrads grads{.query = ttnn::empty_like(query), .key = ttnn::empty_like(key), .value = ttnn::empty_like(value)};
    ttml::metal::ring_sdpa_bw(
        {{.grad_output = tensors.out->get_grad(),
          .attn_output = tensors.out->get_value(),
          .query = query,
          .key = key,
          .value = value,
          .intermediates = global_intermediates,
          .grad_query = grads.query,
          .grad_key = grads.key,
          .grad_value = grads.value}},
        ring.ring_size,
        ring.cp_axis,
        /*step=*/0U,
        mask_type,
        kRingBwDirection);
    return grads;
}

// Adds step 0's grads to the FP32 sum of the other steps' and rounds once to each input's dtype.
void apply_ring_grads(const RingTensors& tensors, const RingGrads& accum, const RingGrads& step0) {
    tensors.query->add_grad(ttnn::typecast(
        ttnn::add(accum.query, ttnn::typecast(step0.query, kFp32)), tensors.query->get_value().dtype()));
    tensors.key->add_grad(
        ttnn::typecast(ttnn::add(accum.key, ttnn::typecast(step0.key, kFp32)), tensors.key->get_value().dtype()));
    tensors.value->add_grad(ttnn::typecast(
        ttnn::add(accum.value, ttnn::typecast(step0.value, kFp32)), tensors.value->get_value().dtype()));
}

// What the contiguous backward keeps from the forward.
struct ContiguousSaved {
    RingTensors tensors;
    ttnn::Tensor lse;
    ttnn::Tensor k_current;
    ttnn::Tensor v_current;
    std::vector<ttnn::Tensor> k_steps;
    std::vector<ttnn::Tensor> v_steps;
    RingSpec ring;
    AttentionMaskType mask_type = AttentionMaskType::Causal;
};

// FP32 sum of the grads of ring steps >= 1. Summing the bf16 kernel outputs in bf16 would round the full running
// magnitude every step.
RingGrads contiguous_backward_ring_steps(ContiguousSaved& saved, const ttnn::Tensor& global_intermediates) {
    using ttnn_fixed::distributed::RingShiftDirection;
    const ttnn::Tensor& query = saved.tensors.query->get_value();
    const ttnn::Tensor& key = saved.tensors.key->get_value();
    const ttnn::Tensor& value = saved.tensors.value->get_value();
    RingGrads accum{
        .query = ttnn::zeros_like(query, kFp32),
        .key = ttnn::zeros_like(key, kFp32),
        .value = ttnn::zeros_like(value, kFp32)};
    const RingGrads zero{
        .query = ttnn::zeros_like(query), .key = ttnn::zeros_like(key), .value = ttnn::zeros_like(value)};
    const RingGrads step_grads{
        .query = ttnn::empty_like(query), .key = ttnn::empty_like(key), .value = ttnn::empty_like(value)};

    const uint32_t cp_axis = saved.ring.cp_axis;
    for (uint32_t step = saved.ring.ring_size - 1U; step >= 1U; --step) {
        const ttnn::Tensor& k = saved.k_steps.empty() ? saved.k_current : saved.k_steps[step];
        const ttnn::Tensor& v = saved.v_steps.empty() ? saved.v_current : saved.v_steps[step];
        // Devices the causal schedule skips at this step do not run the kernels, so their step grads must be zero.
        ttnn::copy(zero.query, step_grads.query);
        ttnn::copy(zero.key, step_grads.key);
        ttnn::copy(zero.value, step_grads.value);
        ttml::metal::ring_sdpa_bw(
            {{.grad_output = saved.tensors.out->get_grad(),
              .attn_output = saved.tensors.out->get_value(),
              .query = query,
              .key = k,
              .value = v,
              .intermediates = global_intermediates,
              .grad_query = step_grads.query,
              .grad_key = step_grads.key,
              .grad_value = step_grads.value}},
            saved.ring.ring_size,
            cp_axis,
            step,
            saved.mask_type,
            kRingBwDirection);
        accum.query = ttnn::add(accum.query, ttnn::typecast(step_grads.query, kFp32));
        accum.key = ttnn::add(accum.key, ttnn::typecast(step_grads.key, kFp32));
        accum.value = ttnn::add(accum.value, ttnn::typecast(step_grads.value, kFp32));

        if (saved.k_steps.empty() && step > 1U) {
            saved.k_current =
                ttnn_fixed::distributed::ring_shift(saved.k_current, cp_axis, RingShiftDirection::Forward);
            saved.v_current =
                ttnn_fixed::distributed::ring_shift(saved.v_current, cp_axis, RingShiftDirection::Forward);
        }
        accum.key = ttnn_fixed::distributed::ring_shift(accum.key, cp_axis, RingShiftDirection::Forward);
        accum.value = ttnn_fixed::distributed::ring_shift(accum.value, cp_axis, RingShiftDirection::Forward);
    }
    return accum;
}

void contiguous_backward(ContiguousSaved& saved) {
    // Standard ring-flash-attention backward: every step gets the UNSCALED upstream gradient, the GLOBAL forward
    // output, and the GLOBAL logsumexp. The sdpa_bw kernels recompute P = exp(scale*QK^T - lse) from the supplied lse,
    // so the global lse yields the global attention weights restricted to this chunk's columns, and
    // u = rowsum(dO * O_global) is the global softmax-backward correction. Per-chunk contributions then sum to the
    // exact dQ/dK/dV. Feeding per-chunk O/lse here would bias dQ/dK (per-chunk u instead of global) — only dV would
    // come out right.
    const ttnn::Tensor global_intermediates = pad_lse_to_intermediates_layout(saved.lse);
    const RingGrads accum = contiguous_backward_ring_steps(saved, global_intermediates);
    const RingGrads step0 = ring_backward_step0(saved.tensors, global_intermediates, saved.ring, saved.mask_type);
    apply_ring_grads(saved.tensors, accum, step0);
}

autograd::TensorPtr ring_attention_sdpa_contiguous(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const RingSpec& ring,
    const AttentionMaskType mask_type,
    const bool reuse_forward_kv) {
    using ttnn_fixed::distributed::RingShiftDirection;
    const auto& query_tensor = query->get_value();
    const uint32_t seq_len_local = query_tensor.logical_shape()[2];
    ttnn::Tensor k_current = key->get_value();
    ttnn::Tensor v_current = value->get_value();

    // Backward at ring step s needs exactly the K/V the forward used at step s. Keeping those handles saves
    // backward its 2 * (ring_size - 2) K/V shifts, at the cost of 2 * (ring_size - 2) extra K-sized tensors
    // per device staying alive until this node's backward has run, or until this op returns when no backward
    // node is registered.
    std::vector<ttnn::Tensor> k_steps(reuse_forward_kv ? ring.ring_size : 0U);
    std::vector<ttnn::Tensor> v_steps(reuse_forward_kv ? ring.ring_size : 0U);

    ttnn::Tensor output_tensor = ttnn::empty_like(query_tensor);
    ttnn::Tensor intermediate_tensor = make_intermediates(query_tensor, seq_len_local);
    const ttnn::Tensor no_contrib_intermediate = make_no_contrib_intermediates(query_tensor, seq_len_local);
    ForwardAccum accum =
        forward_step0(query_tensor, k_current, v_current, ring, mask_type, output_tensor, intermediate_tensor);

    for (uint32_t step = 1U; step < ring.ring_size; ++step) {
        k_current = ttnn_fixed::distributed::ring_shift(k_current, ring.cp_axis, RingShiftDirection::Backward);
        v_current = ttnn_fixed::distributed::ring_shift(v_current, ring.cp_axis, RingShiftDirection::Backward);
        if (reuse_forward_kv) {
            k_steps[step] = k_current;
            v_steps[step] = v_current;
        }
        // Devices the causal schedule skips at this step keep the -inf lse, so the merge ignores this step.
        if (mask_type == AttentionMaskType::Causal) {
            ttnn::copy(no_contrib_intermediate, intermediate_tensor);
        }
        ttml::metal::ring_sdpa_fw(
            {{.query = query_tensor,
              .key = k_current,
              .value = v_current,
              .output = output_tensor,
              .intermediates = intermediate_tensor}},
            ring.ring_size,
            ring.cp_axis,
            step,
            mask_type,
            kRingFwDirection);
        // A device that skips this step re-merges the finite output step 0 wrote, with weight 0.
        online_softmax_merge(accum.output, accum.lse, output_tensor, intermediate_tensor);
    }

    auto out = autograd::create_tensor(ttnn::typecast(accum.output, query_tensor.dtype()));
    if (reuse_forward_kv) {
        // The backward reads k_steps / v_steps; do not hold a second reference to the last step's K/V.
        k_current = ttnn::Tensor();
        v_current = ttnn::Tensor();
    }
    autograd::GradFunction grad_fn = [saved = ContiguousSaved{
                                          .tensors = {.query = query, .key = key, .value = value, .out = out},
                                          .lse = std::move(accum.lse),
                                          .k_current = std::move(k_current),
                                          .v_current = std::move(v_current),
                                          .k_steps = std::move(k_steps),
                                          .v_steps = std::move(v_steps),
                                          .ring = ring,
                                          .mask_type = mask_type}]() mutable { contiguous_backward(saved); };
    out->set_node(autograd::add_backward_node(std::move(grad_fn), out, query, key, value));
    return out;
}

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
    const ForwardAccum step0 = forward_step0(
        query,
        key,
        value,
        ring,
        AttentionMaskType::Causal,
        ttnn::empty_like(query),
        make_intermediates(query, query.logical_shape()[2]));
    return {.output = split_seq(step0.output), .lse = split_seq(step0.lse)};
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
        kRingFwDirection);

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
        kRingFwDirection,
        ttml::metal::ops::RingLoadBalancing::ZigZag);

    online_softmax_merge(accum.output.second, accum.lse.second, buffers.cross_output, buffers.cross_intermediates);
    online_softmax_merge(accum.output.first, accum.lse.first, buffers.chunk_output[0], buffers.chunk_intermediates[0]);
    online_softmax_merge(
        accum.output.second, accum.lse.second, buffers.chunk_output[1], buffers.chunk_intermediates[1]);
}

// What the zig-zag backward keeps from the forward.
struct ZigZagSaved {
    RingTensors tensors;
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
        kRingBwDirection);

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
        kRingBwDirection,
        ttml::metal::ops::RingLoadBalancing::ZigZag);
}

// The cross block's grads go to Q's second half and K/V's first half; chunk c's grads go to half c of Q, K and V.
void accumulate_zigzag_grads(const ZigZagGradBuffers& buffers, ZigZagGradAccum& accum) {
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

void zigzag_backward(ZigZagSaved& saved) {
    using ttnn_fixed::distributed::RingShiftDirection;
    const ttnn::Tensor global_intermediates = pad_lse_to_intermediates_layout(concat_seq(saved.lse));
    const ZigZagBackwardInputs in{
        .query = split_seq(saved.tensors.query->get_value()),
        .attn_output = split_seq(saved.tensors.out->get_value()),
        .grad_output = split_seq(saved.tensors.out->get_grad()),
        .intermediates = split_seq(global_intermediates)};
    ZigZagGradBuffers buffers =
        make_zigzag_grad_buffers(in.query.first, saved.tensors.key->get_value(), saved.tensors.value->get_value());
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
    const RingGrads step0 =
        ring_backward_step0(saved.tensors, global_intermediates, saved.ring, AttentionMaskType::Causal);
    apply_ring_grads(
        saved.tensors,
        {.query = concat_seq(accum.query), .key = concat_seq(accum.key), .value = concat_seq(accum.value)},
        step0);
}

autograd::TensorPtr ring_attention_sdpa_zigzag(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    const RingSpec& ring,
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

    const SeqHalves q_halves = split_seq(query_tensor);
    ZigZagForwardAccum accum = zigzag_forward_step0(query_tensor, key->get_value(), value->get_value(), ring);
    ZigZagForwardBuffers buffers = make_zigzag_forward_buffers(query_tensor, q_halves.first);

    std::vector<SeqHalves> k_steps(reuse_forward_kv ? ring.ring_size : 0U);
    std::vector<SeqHalves> v_steps(reuse_forward_kv ? ring.ring_size : 0U);
    SeqHalves k_current = split_seq(key->get_value());
    SeqHalves v_current = split_seq(value->get_value());
    for (uint32_t step = 1U; step < ring.ring_size; ++step) {
        k_current = ring_shift_halves(k_current, ring.cp_axis, RingShiftDirection::Backward);
        v_current = ring_shift_halves(v_current, ring.cp_axis, RingShiftDirection::Backward);
        if (reuse_forward_kv) {
            k_steps[step] = k_current;
            v_steps[step] = v_current;
        }
        zigzag_forward_step(q_halves, k_current, v_current, step, ring, buffers, accum);
    }

    auto out = autograd::create_tensor(ttnn::typecast(concat_seq(accum.output), query_tensor.dtype()));
    if (reuse_forward_kv) {
        // The backward reads k_steps / v_steps; do not hold a second reference to the last step's K/V.
        k_current = SeqHalves{ttnn::Tensor(), ttnn::Tensor()};
        v_current = SeqHalves{ttnn::Tensor(), ttnn::Tensor()};
    }

    autograd::GradFunction grad_fn = [saved = ZigZagSaved{
                                          .tensors = {.query = query, .key = key, .value = value, .out = out},
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
    const AttentionMaskType mask_type,
    const std::optional<bool> reuse_forward_kv_requested,
    const std::optional<core::distributed::CpSharding> cp_sharding) {
    if (!autograd::ctx().is_parallelism_context_initialized() ||
        !autograd::ctx().get_parallelism_context().is_cp_enabled()) {
        return ttml::ops::scaled_dot_product_attention(query, key, value, mask);
    }

    const auto& pctx = autograd::ctx().get_parallelism_context();
    const RingSpec ring{.ring_size = pctx.get_cp_size(), .cp_axis = pctx.get_cp_axis().value()};
    const bool reuse_forward_kv = reuse_forward_kv_requested.value_or(pctx.get_cp_config().reuse_forward_kv) &&
                                  autograd::ctx().get_gradient_mode() == autograd::GradMode::ENABLED &&
                                  autograd::any_requires_grad(query, key, value);
    TT_FATAL(query->get_value().device() != nullptr, "Query tensor must be on a mesh device for ring attention");
    TT_FATAL(
        !mask.has_value(),
        "Non-causal mask is not supported in CP mode for now, pass nullopt if you want to use causal mask");
    TT_FATAL(
        mask_type != AttentionMaskType::Arbitrary,
        "Arbitrary attention mask is not supported in CP mode, use None or Causal");

    switch (cp_sharding.value_or(pctx.get_cp_config().sharding)) {
        case core::distributed::CpSharding::Contiguous:
            return ring_attention_sdpa_contiguous(query, key, value, ring, mask_type, reuse_forward_kv);
        case core::distributed::CpSharding::ZigZag:
            TT_FATAL(
                mask_type == AttentionMaskType::Causal,
                "Zig-zag CP sharding balances causal attention; use contiguous CP sharding without a causal mask");
            return ring_attention_sdpa_zigzag(query, key, value, ring, reuse_forward_kv);
    }
    TT_THROW("Unknown CP sharding");
}

}  // namespace ttml::ops::distributed
