// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

/**
 * Online-softmax (logsumexp) merge of one ring attention step into running FP32 state, in place and
 * in a single device program:
 *
 *   lse_i       = step_intermediates[..., 0]
 *   new_lse     = logaddexp(running_lse, lse_i)            (stable, via the max)
 *   output_accum = output_accum * exp(running_lse - new_lse) + float(step_output) * exp(lse_i - new_lse)
 *   running_lse = new_lse
 *
 * @param output_accum       FP32 (B, H, S, D) accumulator, updated in place.
 * @param running_lse        FP32 (B, H, S, 32) running logsumexp in the SDPA intermediates layout
 *                           (column 0, zeros elsewhere), updated in place. Start it at -inf in column 0.
 * @param step_output        BF16 (B, H, S, D) normalized attention output of this step.
 * @param step_intermediates FP32 (B, H, S, 32) SDPA intermediates of this step (logsumexp in column 0).
 *
 * A step logsumexp of -inf contributes nothing: the accumulator and running logsumexp keep their values
 * as long as the running logsumexp is finite. All tensors must be TILE, interleaved, in DRAM.
 * output_accum and running_lse are written in place, so they must not share a buffer with any tensor
 * that has to keep its value (ttnn views such as an in-tile ttnn::pad share their input's buffer).
 */
void ring_lse_merge(
    const ttnn::Tensor& output_accum,
    const ttnn::Tensor& running_lse,
    const ttnn::Tensor& step_output,
    const ttnn::Tensor& step_intermediates);

}  // namespace ttml::metal
