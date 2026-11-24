// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::ops::distributed {

// Pads a (B, H, S, 1) FP32 logsumexp tensor into the (B, H, S, 32) intermediates layout
// the SDPA kernels expect: lse in column 0, the remaining 31 columns are ignored padding.
ttnn::Tensor pad_lse_to_intermediates_layout(const ttnn::Tensor& lse);

ttnn::Tensor make_intermediates(const ttnn::Tensor& query, uint32_t seq_len);

// Intermediates whose logsumexp is -inf: merged as a step that contributes nothing, since exp(-inf) = 0.
ttnn::Tensor make_no_contrib_intermediates(const ttnn::Tensor& query, uint32_t seq_len);

ttnn::Tensor slice_seq(
    const ttnn::Tensor& tensor, uint32_t begin, uint32_t end, std::optional<uint32_t> width = std::nullopt);

struct SeqHalves {
    ttnn::Tensor first;
    ttnn::Tensor second;
};

SeqHalves split_seq(const ttnn::Tensor& tensor);

SeqHalves ring_shift_halves(
    const SeqHalves& tensor, uint32_t cp_axis, ttnn_fixed::distributed::RingShiftDirection direction);

/**
 * Online-softmax merge of one ring step's partial attention result.
 *
 * @param output_accum (B, H, S, D) FP32 accumulator, replaced by the merged result
 * @param running_lse (B, H, S, 1) FP32 running logsumexp (-inf = nothing merged yet), replaced by the merged one
 * @param step_output (B, H, S, D) this step's normalized attention output, BFLOAT16. Must be finite in
 *             every row, including rows whose lse is -inf (they are weighted by 0, and 0 * inf = NaN)
 * @param step_intermediates (B, H, S, 32) FP32 SDPA intermediates, logsumexp in column 0
 *             (-inf = the step contributes nothing)
 *
 * Either lse may be -inf in a row, but not both: logaddexp(-inf, -inf) is -inf, so the step weight
 * exp(-inf - (-inf)) is NaN and so is the merged row.
 */
void online_softmax_merge(
    ttnn::Tensor& output_accum,
    ttnn::Tensor& running_lse,
    const ttnn::Tensor& step_output,
    const ttnn::Tensor& step_intermediates);

}  // namespace ttml::ops::distributed
