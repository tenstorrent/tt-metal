// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_sdpa.hpp"

#include <fmt/core.h>

#include <cmath>
#include <limits>
#include <tt-metalium/distributed.hpp>
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
#include "ops/scaled_dot_product_attention.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

namespace {

// Pads a (B, H, S, 1) FP32 logsumexp tensor into the (B, H, S, 32) intermediates layout
// the SDPA kernels expect: lse in column 0, the remaining 31 columns are ignored padding.
ttnn::Tensor pad_lse_to_intermediates_layout(const ttnn::Tensor& lse) {
    const ttsl::SmallVector<ttnn::operations::data_movement::PadSpecDim> padding = {
        {0, 0},  // batch
        {0, 0},  // heads
        {0, 0},  // seq_len
        {0, 31}  // width: pad 31 zeros on the right (1 -> 32)
    };
    return ttnn::pad(lse, padding, 0.0F, false, std::nullopt);
}

}  // namespace

namespace detail {

// Merges one ring step's attention result into the running result (online softmax).
// O_i, l_i: this step's normalized output and logsumexp; O, L: the running ones.
//   L' = logaddexp(L, l_i)
//   w  = exp(l_i - L')            this step's share of the softmax mass, in [0, 1]
//   O' = O + w * (O_i - O)        a lerp, since the usual weight on O, exp(L - L'), is 1 - w
//
// A logsumexp of -inf means no contribution: the first merge (L = -inf) gives O' = O_i, and a
// step the causal schedule skips (l_i = -inf) leaves O and L unchanged. Both -inf would make the
// row NaN (w = exp(-inf - (-inf))); the ring caller never passes that, since every device
// computes its diagonal block at step 0.
// O_i must be finite even on skipped steps (0 * inf = NaN). In the ring caller, step 0 writes
// every row of the reused output buffer on every device, so a skipped step re-merges an
// earlier finite output with weight 0.
void online_softmax_merge(
    ttnn::Tensor& output_accum,
    ttnn::Tensor& running_lse,
    const ttnn::Tensor& step_output,
    const ttnn::Tensor& step_intermediates) {
    TT_FATAL(
        output_accum.dtype() == ttnn::DataType::FLOAT32 && running_lse.dtype() == ttnn::DataType::FLOAT32 &&
            step_intermediates.dtype() == ttnn::DataType::FLOAT32,
        "online_softmax_merge: output_accum, running_lse and step_intermediates must be FLOAT32, got {}, {}, {}",
        output_accum.dtype(),
        running_lse.dtype(),
        step_intermediates.dtype());
    TT_FATAL(
        step_output.dtype() == ttnn::DataType::BFLOAT16 || step_output.dtype() == ttnn::DataType::FLOAT32,
        "online_softmax_merge: step_output must be BFLOAT16 or FLOAT32, got {}",
        step_output.dtype());
    TT_FATAL(
        step_intermediates.logical_shape()[-1] == 32U,
        "online_softmax_merge: step_intermediates must be (B, H, S, 32), got {}",
        step_intermediates.logical_shape());

    // 1. lse of this step: column 0 of the (B, H, S, 32) FP32 intermediates, as a (B, H, S, 1)
    //    view of the same tiles. Columns 1..31 become tile padding. No copy and no dispatch: this
    //    is the tensor ttnn::slice would return, without slice's copy of every tile.
    const auto [batch, heads, seq_len, width] = step_intermediates.logical_shape().to_array_4D();
    const ttnn::Tensor lse_chunk = ttnn::experimental::view(
        step_intermediates, ttnn::Shape{batch, heads, seq_len, 1U}, step_intermediates.padded_shape());

    // 2. L' = logaddexp(L, l_i). A single fused SFPU kernel evaluating the stable form
    //    max(a, b) + log1p(exp(-|a - b|)) (no overflow for large lse), with
    //    logaddexp(-inf, x) = x.
    ttnn::Tensor new_lse = ttnn::logaddexp(running_lse, lse_chunk);

    // 3. w = exp(l_i - L'): subtract with exp fused as a post-activation (one kernel; 0.0f
    //    selects the accurate exp, ttnn::exp's default).
    using EltwiseUnary = ttnn::operations::unary::EltwiseUnaryWithParam;
    const EltwiseUnary exp_act{ttnn::operations::unary::UnaryOpType::EXP, 0.0F};
    const ttnn::Tensor new_weight = ttnn::subtract(
        lse_chunk,
        new_lse,
        /*output_dtype=*/std::nullopt,
        /*memory_config=*/std::nullopt,
        /*output=*/std::nullopt,
        /*post_activations=*/ttsl::Span<const EltwiseUnary>(&exp_act, 1));

    // 4. O' = O + w * (O_i - O). The ternary lerp kernel broadcasts the (B, H, S, 1) weight
    //    along the head dim and takes the bf16 step output as is: every operand is unpacked
    //    into an FP32 Dest (bf16 -> FP32 is exact) and the output takes the accumulator's FP32
    //    dtype, so the whole combine stays FP32 without a separate typecast. The FP32
    //    accumulator matters: backward consumes the saved O in u = rowsum(dO * O).
    output_accum = ttnn::lerp(output_accum, step_output, new_weight);
    running_lse = std::move(new_lse);
}

}  // namespace detail

namespace {

ttnn::Tensor to_fp32(const ttnn::Tensor& tensor) {
    return ttnn::typecast(tensor, ttnn::DataType::FLOAT32);
}

// Shifts a (B, H, S, X) TILE tensor by one row along the sequence dimension:
//   up:   out[r] = in[r + 1] for r < S - 1, out[S - 1] = fill
//   down: out[r] = in[r - 1] for r > 0,     out[0]     = fill
// A one-row shift is not tile aligned, so it goes through row-major. Striped ring attention
// calls it a fixed number of times per layer (never per ring step); it moves O(S * X) data
// against the O(S^2 * X) of the attention it enables.
enum class SeqShift : uint8_t { Up, Down };

ttnn::Tensor shift_seq_one_row(const ttnn::Tensor& tensor, SeqShift direction, float fill) {
    const auto [batch, heads, seq_len, width] = tensor.logical_shape().to_array_4D();
    TT_FATAL(seq_len >= 2U, "Cannot shift a sequence of length {} by one row", seq_len);
    TT_FATAL(tensor.device() != nullptr, "shift_seq_one_row expects a device tensor");

    // Untilize, drop the row that falls off, pad the opposite end with one fill row, retilize.
    // Row-major pad accepts front padding on any dim and fills on device (no host upload).
    const ttnn::Tensor row_major = ttnn::to_layout(tensor, ttnn::Layout::ROW_MAJOR);
    const bool up = direction == SeqShift::Up;
    const ttsl::SmallVector<uint32_t> start = {0, 0, up ? 1U : 0U, 0};
    const ttsl::SmallVector<uint32_t> stop = {batch, heads, up ? seq_len : seq_len - 1U, width};
    const ttsl::SmallVector<uint32_t> step = {1, 1, 1, 1};
    const ttnn::Tensor kept = ttnn::slice(row_major, start, stop, step);
    const ttsl::SmallVector<ttnn::operations::data_movement::PadSpecDim> padding = {
        {0, 0},
        {0, 0},
        {up ? 0U : 1U, up ? 1U : 0U},  // sequence: one fill row after (up) or before (down)
        {0, 0}};
    const ttnn::Tensor shifted = ttnn::pad(kept, padding, fill);
    return ttnn::to_layout(shifted, ttnn::Layout::TILE);
}

// Striped (load-balanced causal) ring attention. Global token p lives on rank p mod N as
// local row p div N, so rank d's row l is global l*N + d. For the K/V of rank src, key m is
// visible to query l iff m*N + src <= l*N + d:
//   src <= d: m <= l, the kernel's causal mask on Q;
//   src >  d: m <  l, strict causal, computed as causal on Q' = shift_up(Q): kernel row r is
//             query r + 1 and sees m <= r = (r + 1) - 1. These results are in shifted row
//             coordinates; the last Q' row is zero (junk), real row 0 sees nothing.
// Every rank runs one causal L x L block of the same shape at every step (no rank idles), so
// the critical path is N causal blocks instead of one causal plus N-1 full blocks.
//
// Results on Q and on Q' are merged into two FP32 accumulators (real and shifted rows); the
// shifted one is moved back one row and merged into the real one once at the end. The backward
// mirrors this: the Q' devices get the shifted dO, O and logsumexp too, dQ is accumulated per
// query and realigned once, and dK/dV are in source coordinates either way and travel the ring
// as in the contiguous layout.
autograd::TensorPtr ring_attention_sdpa_striped(
    const autograd::TensorPtr& query,
    const autograd::TensorPtr& key,
    const autograd::TensorPtr& value,
    uint32_t ring_size,
    uint32_t cp_axis_value,
    tt::tt_metal::distributed::MeshDevice* mesh_device) {
    constexpr auto kRingDirection = ttml::metal::ops::ring_sdpa_fw::RingDirection::Backward;
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    // Running logsumexp of the shifted accumulator before anything was merged into it. It must
    // be finite: rank N-1 never uses Q', and the merge of two -inf logsumexps is NaN.
    // exp(kLseFloor - x) is 0 for any real logsumexp x, so it never weighs in.
    constexpr float kLseFloor = -1.0e30F;

    const auto& query_tensor = query->get_value();
    const auto [batch_num, heads, seq_len_local, dim] = query_tensor.logical_shape().to_array_4D();

    ttnn::Tensor k_current = key->get_value();
    ttnn::Tensor v_current = value->get_value();
    const ttnn::Tensor query_shifted = shift_seq_one_row(query_tensor, SeqShift::Up, 0.0F);

    const auto dram = ttnn::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM);
    auto make_intermediates = [&]() {
        return ttnn::empty(
            ttnn::Shape{batch_num, heads, seq_len_local, 32U},
            ttnn::DataType::FLOAT32,
            ttnn::Layout::TILE,
            mesh_device,
            dram);
    };
    // Step buffers, reused every step: [0] for results on Q, [1] for results on Q' (shifted
    // rows). Each rank writes exactly one of them per step; the other one's intermediates are
    // reset to "no contribution" (-inf) and its output buffer must stay finite for the merge
    // (weight 0 times a finite value), hence zeros_like.
    ttnn::Tensor step_output[2] = {ttnn::zeros_like(query_tensor), ttnn::zeros_like(query_tensor)};
    ttnn::Tensor step_intermediates[2] = {make_intermediates(), make_intermediates()};
    const ttnn::Tensor no_contrib_intermediates = pad_lse_to_intermediates_layout(ttnn::full(
        ttnn::Shape{batch_num, heads, seq_len_local, 1U},
        kNegInf,
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        std::ref(*mesh_device)));

    // FP32 accumulators: [0] in real row coordinates, [1] in shifted ones.
    ttnn::Tensor accum[2] = {
        ttnn::zeros_like(query_tensor, ttnn::DataType::FLOAT32),
        ttnn::zeros_like(query_tensor, ttnn::DataType::FLOAT32)};
    ttnn::Tensor running_lse[2] = {
        ttnn::full(
            ttnn::Shape{batch_num, heads, seq_len_local, 1U},
            kNegInf,
            ttnn::DataType::FLOAT32,
            ttnn::Layout::TILE,
            std::ref(*mesh_device)),
        ttnn::full(
            ttnn::Shape{batch_num, heads, seq_len_local, 1U},
            kLseFloor,
            ttnn::DataType::FLOAT32,
            ttnn::Layout::TILE,
            std::ref(*mesh_device))};

    for (uint32_t step = 0; step < ring_size; ++step) {
        if (step > 0U) {
            k_current = ttnn_fixed::distributed::ring_shift(
                k_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Backward);
            v_current = ttnn_fixed::distributed::ring_shift(
                v_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Backward);
            // From step 1 on, ranks split between Q and Q' (src = d + step > d iff d < N - step).
            ttnn::copy(no_contrib_intermediates, step_intermediates[0]);
            ttnn::copy(no_contrib_intermediates, step_intermediates[1]);
        }
        // At step 0 every rank holds its own K/V (src == d) and runs on Q, which writes every
        // row of the [0] buffers; the [1] buffers are not touched and not merged.
        ttml::metal::ring_sdpa_fw(
            query_tensor,
            k_current,
            v_current,
            ring_size,
            cp_axis_value,
            step,
            detail::kStripedOnQuery,
            kRingDirection,
            step_output[0],
            step_intermediates[0]);
        if (step > 0U) {
            ttml::metal::ring_sdpa_fw(
                query_shifted,
                k_current,
                v_current,
                ring_size,
                cp_axis_value,
                step,
                detail::kStripedOnShiftedQuery,
                kRingDirection,
                step_output[1],
                step_intermediates[1]);
        }

        detail::merge_partial_into(accum[0], running_lse[0], step_output[0], step_intermediates[0]);
        if (step > 0U) {
            detail::merge_partial_into(accum[1], running_lse[1], step_output[1], step_intermediates[1]);
        }
    }

    // Move the shifted accumulator back to real rows (row r -> r + 1; its last row is the junk
    // row of Q' and drops out, real row 0 gets "no contribution") and merge it into the real one.
    const ttnn::Tensor shifted_back_output = shift_seq_one_row(accum[1], SeqShift::Down, 0.0F);
    // Fill with the finite floor, not -inf: row-major ttnn::pad packs its fill value as a pair of bf16
    // halves even for FP32 tensors, so -inf would come out as 0xFF80FF80 (NaN) and poison real row 0.
    // kLseFloor survives that packing as a huge negative finite value, which merges with weight 0.
    const ttnn::Tensor shifted_back_intermediates =
        shift_seq_one_row(pad_lse_to_intermediates_layout(running_lse[1]), SeqShift::Down, kLseFloor);
    detail::merge_partial_into(accum[0], running_lse[0], shifted_back_output, shifted_back_intermediates);

    auto out = autograd::create_tensor(ttnn::typecast(accum[0], query_tensor.dtype()));
    const ttnn::Tensor final_lse = running_lse[0];

    autograd::GradFunction grad_fn = [query,
                                      key,
                                      value,
                                      out,
                                      final_lse,
                                      k_current,  // K at the end of the forward ring
                                      v_current,  // V at the end of the forward ring
                                      ring_size,
                                      cp_axis_value,
                                      mesh_device]() mutable {
        tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());
        const ttnn::Tensor& grad_output = out->get_grad();
        const ttnn::Tensor& attn_output = out->get_value();
        const ttnn::Tensor& query_tensor = query->get_value();

        // Global O / logsumexp (see the contiguous backward for why every block must use them).
        // The Q' devices get them shifted up one row like their query. The junk last row is zero in dO',
        // O', Q' and the logsumexp: there P = exp(0 - 0) = 1 is finite, dP = dO' V^T = 0 and
        // u = rowsum(dO' * O') = 0, so dS = 0 and the row adds nothing to dK/dV (dV += P^T dO' = 0).
        const ttnn::Tensor global_intermediates = pad_lse_to_intermediates_layout(final_lse);
        const ttnn::Tensor query_shifted = shift_seq_one_row(query_tensor, SeqShift::Up, 0.0F);
        const ttnn::Tensor attn_output_shifted = shift_seq_one_row(attn_output, SeqShift::Up, 0.0F);
        const ttnn::Tensor grad_output_shifted = shift_seq_one_row(grad_output, SeqShift::Up, 0.0F);
        const ttnn::Tensor intermediates_shifted = shift_seq_one_row(global_intermediates, SeqShift::Up, 0.0F);

        // dQ per query (real and shifted rows); dK/dV are in the K/V source's coordinates either
        // way, so both share the dK/dV step buffers. Every rank runs every step and
        // the causal bw_kv kernel writes every K/V row, so those buffers need no reset; the dQ
        // step buffers do, since each rank writes only one of them per step.
        ttnn::Tensor grad_Q_accum[2] = {
            ttnn::zeros_like(query_tensor, ttnn::DataType::FLOAT32),
            ttnn::zeros_like(query_tensor, ttnn::DataType::FLOAT32)};
        ttnn::Tensor grad_K_accum = ttnn::zeros_like(key->get_value(), ttnn::DataType::FLOAT32);
        ttnn::Tensor grad_V_accum = ttnn::zeros_like(value->get_value(), ttnn::DataType::FLOAT32);
        ttnn::Tensor grad_Q_step[2] = {ttnn::zeros_like(query_tensor), ttnn::zeros_like(query_tensor)};
        ttnn::Tensor grad_K_step = ttnn::zeros_like(key->get_value());
        ttnn::Tensor grad_V_step = ttnn::zeros_like(value->get_value());
        const ttnn::Tensor zero_Q = ttnn::zeros_like(query_tensor);

        // Reverse ring, as in the contiguous backward: K/V and the travelling dK/dV accumulators
        // shift in the Forward direction; the schedule uses the same src = (d + step) mod N.
        for (int step = static_cast<int>(ring_size) - 1; step >= 0; --step) {
            const auto step_idx = static_cast<uint32_t>(step);
            // Step 0 runs on Q on every rank, which writes all of grad_Q_step[0].
            if (step_idx > 0U) {
                ttnn::copy(zero_Q, grad_Q_step[0]);
                ttnn::copy(zero_Q, grad_Q_step[1]);
            }
            ttml::metal::ring_sdpa_bw(
                grad_output,
                attn_output,
                query_tensor,
                k_current,
                v_current,
                global_intermediates,
                ring_size,
                cp_axis_value,
                step_idx,
                detail::kStripedOnQuery,
                ttml::metal::ops::ring_sdpa_bw::RingDirection::Backward,
                grad_Q_step[0],
                grad_K_step,
                grad_V_step);
            if (step_idx > 0U) {
                // Runs on the devices the call above skipped, so they share the dK/dV step buffers.
                ttml::metal::ring_sdpa_bw(
                    grad_output_shifted,
                    attn_output_shifted,
                    query_shifted,
                    k_current,
                    v_current,
                    intermediates_shifted,
                    ring_size,
                    cp_axis_value,
                    step_idx,
                    detail::kStripedOnShiftedQuery,
                    ttml::metal::ops::ring_sdpa_bw::RingDirection::Backward,
                    grad_Q_step[1],
                    grad_K_step,
                    grad_V_step);
            }

            grad_Q_accum[0] = ttnn::add(grad_Q_accum[0], to_fp32(grad_Q_step[0]));
            if (step_idx > 0U) {
                grad_Q_accum[1] = ttnn::add(grad_Q_accum[1], to_fp32(grad_Q_step[1]));
            }
            grad_K_accum = ttnn::add(grad_K_accum, to_fp32(grad_K_step));
            grad_V_accum = ttnn::add(grad_V_accum, to_fp32(grad_V_step));

            if (step_idx > 0U) {
                k_current = ttnn_fixed::distributed::ring_shift(
                    k_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
                v_current = ttnn_fixed::distributed::ring_shift(
                    v_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
                grad_K_accum = ttnn_fixed::distributed::ring_shift(
                    grad_K_accum, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
                grad_V_accum = ttnn_fixed::distributed::ring_shift(
                    grad_V_accum, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
            }
        }

        // Shifted-row dQ back to real rows: row r is query r + 1 (the junk last row drops out,
        // real row 0 gets nothing from Q').
        const ttnn::Tensor grad_Q =
            ttnn::add(grad_Q_accum[0], shift_seq_one_row(grad_Q_accum[1], SeqShift::Down, 0.0F));
        query->add_grad(ttnn::typecast(grad_Q, query_tensor.dtype()));
        key->add_grad(ttnn::typecast(grad_K_accum, key->get_value().dtype()));
        value->add_grad(ttnn::typecast(grad_V_accum, value->get_value().dtype()));
    };

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
    RingAttentionLayout layout) {
    if (!autograd::ctx().is_parallelism_context_initialized() ||
        !autograd::ctx().get_parallelism_context().is_cp_enabled()) {
        return ttml::ops::scaled_dot_product_attention(query, key, value, mask);
    }

    const auto& pctx = autograd::ctx().get_parallelism_context();
    const uint32_t cp_axis_value = pctx.get_cp_axis().value();
    const uint32_t ring_size = pctx.get_cp_size();
    const auto& query_tensor = query->get_value();
    auto* mesh_device = query_tensor.device();
    TT_FATAL(mesh_device != nullptr, "Query tensor must be on a mesh device for ring attention");
    TT_FATAL(
        !mask.has_value(),
        "Non-causal mask is not supported in CP mode for now, pass nullopt if you want to use causal mask");
    TT_FATAL(
        mask_type != ttml::metal::AttentionMaskType::Arbitrary,
        "Arbitrary attention mask is not supported in CP mode, use None or Causal");
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());

    if (layout == RingAttentionLayout::Striped && ring_size > 1U) {
        TT_FATAL(
            mask_type == ttml::metal::AttentionMaskType::Causal,
            "Striped ring attention load-balances causal attention; use the contiguous layout without a causal mask");
        return ring_attention_sdpa_striped(query, key, value, ring_size, cp_axis_value, mesh_device);
    }

    auto [batch_num, heads, seq_len_local, dim] = query_tensor.logical_shape().to_array_4D();
    // Initialize current K and V (will be ring-shifted each step)
    // Use raw ttnn::Tensor since we define backward manually
    ttnn::Tensor k_current = key->get_value();
    ttnn::Tensor v_current = value->get_value();

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
    ttnn::Tensor intermediate_tensor = ttnn::empty(
        ttnn::Shape{batch_num, heads, seq_len_local, 32U},
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        mesh_device,
        ttnn::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM));

    // "no contribution" intermediate: logsumexp = -inf (col 0), rest zeros
    // exp(-inf) = 0, so this chunk contributes nothing to the combined softmax
    ttnn::Tensor col0_neg_inf = ttnn::full(
        ttnn::Shape{batch_num, heads, seq_len_local, 1U},
        -std::numeric_limits<float>::infinity(),
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        std::ref(*mesh_device));
    ttnn::Tensor no_contrib_intermediate = pad_lse_to_intermediates_layout(col0_neg_inf);

    for (uint32_t step = 0; step < ring_size; ++step) {
        // For causal masking, initialize intermediate_tensor to "no contribution" values
        // Devices that are skipped will keep these values, indicating zero contribution
        if (mask_type == ttml::metal::AttentionMaskType::Causal) {
            ttnn::copy(no_contrib_intermediate, intermediate_tensor);
        }

        auto [out_tensor, inter_tensor] = ttml::metal::ring_sdpa_fw(
            query_tensor,
            k_current,
            v_current,
            ring_size,
            cp_axis_value,
            step,
            ttml::metal::ops::contiguous_ring_step_masks(mask_type),
            ttml::metal::ops::ring_sdpa_fw::RingDirection::Backward,
            output_tensor,
            intermediate_tensor);

        // Fold this step's partial result into the FP32 accumulator and the running lse.
        detail::online_softmax_merge(output_accum, global_lse, output_tensor, intermediate_tensor);

        if (step < ring_size - 1) {
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
                                      ring_size,
                                      cp_axis_value,
                                      mask_type,
                                      mesh_device]() mutable {
        tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, std::vector<tt::tt_metal::SubDeviceId>());
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
            auto [grad_Q_result, grad_K_result, grad_V_result] = ttml::metal::ring_sdpa_bw(
                grad_output,
                attn_output,  // Global forward output saved by autograd
                query_tensor,
                k_current,             // K at current ring position
                v_current,             // V at current ring position
                global_intermediates,  // Global logsumexp in intermediates layout
                ring_size,
                cp_axis_value,
                step_idx,
                ttml::metal::ops::contiguous_ring_step_masks(mask_type),
                ttml::metal::ops::ring_sdpa_bw::RingDirection::Backward,
                grad_Q_step,
                grad_K_step,
                grad_V_step);

            // The results alias the preallocated step buffers; skipped devices keep the
            // zeros written above. Upcast so the accumulation stays FP32.
            grad_Q_accum = ttnn::add(grad_Q_accum, ttnn::typecast(grad_Q_result, ttnn::DataType::FLOAT32));
            grad_K_accum = ttnn::add(grad_K_accum, ttnn::typecast(grad_K_result, ttnn::DataType::FLOAT32));
            grad_V_accum = ttnn::add(grad_V_accum, ttnn::typecast(grad_V_result, ttnn::DataType::FLOAT32));

            // Ring shift K/V and grad accumulators in FORWARD direction
            // K/V: replays the forward pass in reverse (gets K/V for previous step)
            // grad_K/V: routes accumulated gradients back to correct device
            if (step > 0) {
                // Shift K/V forward to get position for next backward iteration
                k_current = ttnn_fixed::distributed::ring_shift(
                    k_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);
                v_current = ttnn_fixed::distributed::ring_shift(
                    v_current, cp_axis_value, ttnn_fixed::distributed::RingShiftDirection::Forward);

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
