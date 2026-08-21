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
 * g_v = -log(-log(U_v)) and U ~ Uniform, in ONE device op. When spelled out as a composed
 * implementation, it would run eight ttnn ops and stage several [B, 1, tokens, V] tensors through DRAM;
 * this fused op streams a couple of tiles at a time and reduces on the fly, so peak L1/DRAM for the
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
 *                    noise everywhere.
 * @param logits_mask Optional additive mask, same dtype as `logits`, SUBTRACTED from the scores
 *                    after temperature scaling: score = logits / T + g - mask. Two shapes:
 *                      [1, 1, 1, V] -- one row shared by every batch entry (vocab-padding masking);
 *                      [B, 1, 1, V] -- one row PER batch entry (general per-request logit bias:
 *                                      banned ids, OpenAI-style logit_bias, repetition penalties),
 *                                      broadcast down token positions. On a mesh, shard it with the
 *                                      SAME mapper as the batch, exactly like `positions`.
 *                    Unit note: in the SAMPLED path the mask lands post-scaling
 *                    (score = logits / T + g - mask), so a finite bias b here shifts the RAW logits
 *                    by b * T -- pass mask = -bias / T for exact pre-temperature logit_bias
 *                    semantics. In the GREEDY path (temperature 0, or below the reciprocal-overflow
 *                    floor) there is no scaling (score = logits - mask), so pass mask = -bias
 *                    directly. For +-1e4-style masking the distinction is immaterial either way.
 * @param positions   Optional per-batch-entry token position: [B, 1, 1, 1] UINT32 ROW_MAJOR
 *                    INTERLEAVED, i.e. this op's own position-mode output spec. When absent, samples every
 *                    position. When supplied, ONLY row positions[b] of batch entry b is read,
 *                    reduced and written, and the result is [B, 1, 1, 1]. Shard it with the SAME
 *                    mapper the batch was sharded with; the op does no global->local mapping.
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
    const std::optional<ttnn::Tensor>& logits_mask = std::nullopt,
    const std::optional<ttnn::Tensor>& positions = std::nullopt);

}  // namespace ttml::metal
