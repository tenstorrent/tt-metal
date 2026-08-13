// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

/**
 * Single-kernel Gumbel-max sampling.
 *
 * Computes argmax_v( logits[..., v] / temperature + g_v  [- mask[..., v]] ), with
 * g_v = -log(-log(U_v)) and U ~ Uniform, in ONE device op. The composed spelling in
 * ttnn_fixed::sample runs eight ttnn ops and stages several [B, 1, tokens, V] tensors through DRAM;
 * this streams a couple of tiles at a time and reduces on the fly, so peak L1/DRAM for the
 * intermediates is O(1) in V rather than O(B * tokens * V).
 *
 * @param logits      TILE layout, [B, 1, tokens, V], BFLOAT16 or FLOAT32.
 * @param temperature >= 0. Zero means greedy: the noise and the scaling are compiled out and the op
 *                    reduces to an argmax over the (masked) logits -- still fused, so the greedy
 *                    path also avoids the untilized full-size copy ttnn::argmax would need.
 * @param seed        Any value, including 0. Unlike ttnn::rand, 0 is not a sentinel for host
 *                    entropy -- it is just a seed, and sampling is always reproducible for a given
 *                    (seed, mesh shape, work split). Callers that want non-reproducible sampling
 *                    must vary the seed themselves.
 * @param seed_axes   Mesh axes that must draw DISTINCT noise per device (the data-parallel axes).
 *                    Axes omitted stay in lockstep across devices. Empty (default) => identical
 *                    noise everywhere, matching ttnn_fixed::sample's default.
 * @param logits_padding_mask Optional additive mask, same shape/dtype as `logits`, subtracted from
 *                    the scores.
 * @param positions   Optional token position to sample for each batch row. Empty (default) samples
 *                    every position. When supplied, ONLY row positions[b] of batch entry b is read,
 *                    reduced and written, and the result is [B, 1, 1, 1] instead of
 *                    [B, 1, tokens, 1]. Prefill wants exactly that -- one token per sequence, at
 *                    that sequence's own prompt end -- and gets an Ht-fold cut in work for it. Pass
 *                    either this device's own rows (B_local entries) or the whole job's list, which
 *                    is sharded across the `seed_axes` the same way the batch is.
 *
 * @return ROW_MAJOR UINT32 token ids, [B, 1, tokens, 1] -- identical in shape, dtype and layout to
 *         what ttnn::argmax(..., dim=3, keepdim=true) returns today -- or [B, 1, 1, 1] when
 *         `positions` is supplied.
 */
ttnn::Tensor gumbel_sample(
    const ttnn::Tensor& logits,
    float temperature,
    uint32_t seed,
    const std::vector<uint32_t>& seed_axes = {},
    const std::optional<ttnn::Tensor>& logits_padding_mask = std::nullopt,
    const std::vector<uint32_t>& positions = {});

}  // namespace ttml::metal
