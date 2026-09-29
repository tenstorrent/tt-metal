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
 * Single-kernel Gumbel-max sampling: argmax_v( logits[..., v] / temperature + g_v [- mask[..., v]] )
 * with g_v = -log(-log(U_v)), U ~ Uniform, in one device op. Streams a few tiles at a time and
 * reduces on the fly, so peak L1/DRAM for intermediates is O(1) in V.
 *
 * @param logits      TILE layout, [B, 1, tokens, V], BFLOAT16 or FLOAT32.
 * @param temperature >= 0. Zero (or a positive value whose reciprocal overflows float32) selects
 *                    greedy: noise and scaling are compiled out, leaving a fused argmax.
 * @param seed        Any value, including 0 (no host-entropy sentinel, unlike ttnn::rand).
 *                    Sampling is reproducible for a given (seed, mesh shape, work split).
 * @param seed_axes   Mesh axes that draw DISTINCT noise per device (the data-parallel axes).
 *                    Omitted axes stay in lockstep. Empty (default) => identical noise everywhere.
 * @param logits_mask Optional additive mask, same dtype as `logits`, subtracted from the scores:
 *                    [1, 1, 1, V] (shared vocab padding) or [B, 1, 1, V] (per-entry logit bias),
 *                    broadcast down token rows. Shard per-row masks like `positions`. The subtract
 *                    lands post-scaling, so for exact pre-temperature logit_bias semantics pass
 *                    mask = -bias / T in the sampled path (mask = -bias in greedy); for
 *                    +-1e4-style masking the distinction is immaterial.
 * @param positions   Optional per-batch-entry token position: [B, 1, 1, 1] UINT32 ROW_MAJOR
 *                    INTERLEAVED (this op's own position-mode output spec). When supplied, only
 *                    row positions[b] of entry b is read, reduced and written. Shard it with the
 *                    same mapper as the batch; the op does no global->local mapping.
 *
 * @return ROW_MAJOR UINT32 token ids, [B, 1, tokens, 1] -- matching ttnn::argmax(dim=3, keepdim)
 *         -- or [B, 1, 1, 1] when `positions` is supplied.
 */
ttnn::Tensor gumbel_sample(
    const ttnn::Tensor& logits,
    float temperature,
    uint32_t seed,
    const std::vector<uint32_t>& seed_axes = {},
    const std::optional<ttnn::Tensor>& logits_mask = std::nullopt,
    const std::optional<ttnn::Tensor>& positions = std::nullopt);

}  // namespace ttml::metal
