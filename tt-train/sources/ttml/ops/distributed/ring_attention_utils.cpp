// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_utils.hpp"

#include <functional>
#include <limits>

#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"

namespace ttml::ops::distributed {

ttnn::Tensor pad_lse_to_intermediates_layout(const ttnn::Tensor& lse) {
    const ttsl::SmallVector<ttnn::operations::data_movement::PadSpecDim> padding = {
        {0, 0},  // batch
        {0, 0},  // heads
        {0, 0},  // seq_len
        {0, 31}  // width: pad 31 zeros on the right (1 -> 32)
    };
    return ttnn::pad(lse, padding, 0.0F, false, std::nullopt);
}

ttnn::Tensor slice_seq(const ttnn::Tensor& tensor, uint32_t begin, uint32_t end, std::optional<uint32_t> width) {
    const auto [batch, heads, seq_len, tensor_width] = tensor.logical_shape().to_array_4D();
    TT_FATAL(begin < end && end <= seq_len, "Invalid sequence slice [{}, {}) of length {}", begin, end, seq_len);
    const ttsl::SmallVector<uint32_t> start = {0U, 0U, begin, 0U};
    const ttsl::SmallVector<uint32_t> stop = {batch, heads, end, width.value_or(tensor_width)};
    const ttsl::SmallVector<uint32_t> step = {1U, 1U, 1U, 1U};
    return ttnn::slice(tensor, start, stop, step);
}

SeqHalves split_seq(const ttnn::Tensor& tensor) {
    const uint32_t seq_len = tensor.logical_shape()[2];
    return {slice_seq(tensor, 0U, seq_len / 2U), slice_seq(tensor, seq_len / 2U, seq_len)};
}

SeqHalves ring_shift_halves(
    const SeqHalves& tensor, uint32_t cp_axis, ttnn_fixed::distributed::RingShiftDirection direction) {
    return {
        ttnn_fixed::distributed::ring_shift(tensor.first, cp_axis, direction),
        ttnn_fixed::distributed::ring_shift(tensor.second, cp_axis, direction)};
}

ttnn::Tensor make_intermediates(const ttnn::Tensor& query, uint32_t seq_len) {
    const auto& shape = query.logical_shape();
    return ttnn::empty(
        ttnn::Shape{shape[0], shape[1], seq_len, 32U},
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        query.device(),
        ttnn::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, ttnn::BufferType::DRAM));
}

ttnn::Tensor make_no_contrib_intermediates(const ttnn::Tensor& query, uint32_t seq_len) {
    const auto& shape = query.logical_shape();
    return pad_lse_to_intermediates_layout(ttnn::full(
        ttnn::Shape{shape[0], shape[1], seq_len, 1U},
        -std::numeric_limits<float>::infinity(),
        ttnn::DataType::FLOAT32,
        ttnn::Layout::TILE,
        std::ref(*query.device())));
}

// Merges one ring step's attention result into the running result (online softmax).
// O_i, l_i: this step's normalized output and logsumexp; O, L: the running ones.
//   L' = logaddexp(L, l_i)
//   w  = exp(l_i - L')            this step's share of the softmax mass, in [0, 1]
//   O' = O + w * (O_i - O)        a lerp, since the usual weight on O, exp(L - L'), is 1 - w
//
// A logsumexp of -inf means no contribution: the first merge (L = -inf) gives O' = O_i, and a
// step the causal schedule skips (l_i = -inf) leaves O and L unchanged. Both -inf would make the
// row NaN (w = exp(-inf - (-inf))). O_i must be finite even on skipped steps (0 * inf = NaN).
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
        step_output.dtype() == ttnn::DataType::BFLOAT16,
        "online_softmax_merge: step_output must be BFLOAT16, got {}",
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
    //    lerp is not overflow-safe: it forms O_i - O, which overflows FP32 once |O| and |O_i|
    //    approach FLT_MAX/2, even with both finite (NaN at w = 0). Tracked in
    //    https://github.com/tenstorrent/tt-metal/issues/65873.
    output_accum = ttnn::lerp(output_accum, step_output, new_weight);
    running_lse = std::move(new_lse);
}

}  // namespace ttml::ops::distributed
