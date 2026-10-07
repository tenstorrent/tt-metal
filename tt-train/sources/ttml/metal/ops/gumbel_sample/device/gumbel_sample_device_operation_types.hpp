// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gumbel_sample::device {

// Whether this temperature selects the Gumbel-noise kernel variant or greedy argmax.
// Deliberately STRICTER than `temperature > 0`: the device receives 1/temperature as a runtime
// arg, and for positive temperatures below ~1/FLT_MAX (~2.9e-39, subnormal range) that reciprocal
// overflows to +inf. logit * inf then collapses every positive logit to the same +inf bit pattern
// and every zero logit to NaN (which float32_greater never picks), so the "sampled" argmax would
// silently return the FIRST positive column rather than the max one. A temperature that small
// means greedy anyway -- the noise term is dwarfed by the scaled logits long before the
// reciprocal overflows -- so it routes to the greedy kernel, which is the exact limit a
// temperature anneal is approaching.
//
// Single-sourced here because three sites must agree on which kernel a temperature selects: the
// program HASH, the kernel-define selection in the factory, and the runtime-arg override. A
// disagreement would replay a cached greedy program with noise args, or vice versa.
inline bool uses_gumbel_noise(float temperature) {
    return temperature > 0.0F && std::isfinite(1.0F / temperature);
}

struct GumbelSampleParams {
    // Softmax temperature, >= 0. Zero -- or a positive value so small that 1/temperature
    // overflows float32 (see uses_gumbel_noise above) -- selects greedy argmax: the compute kernel
    // compiles out the RNG and the scaling (the noise compile-time arg) and passes the logits straight to the
    // writer's running argmax. Note this makes uses_gumbel_noise(temperature) part of the program
    // hash, since it changes the kernel binary -- see compute_program_hash.
    float temperature = 1.0F;

    // RNG seed. Any value is valid, INCLUDING 0 -- it is an ordinary seed here, not a sentinel.
    // This op seeds the SFPU generator directly (rand_tile_init) instead of going through
    // ttnn::rand, so it does not inherit that op's "seed == 0 => host time-based entropy"
    // contract; sampling here is always reproducible for a given seed. Zero is safe because the
    // hardware generator is an XNOR LFSR whose sole lock-up state is all-ones, which
    // ckernel_sfpu_rand.h already rewrites.
    uint32_t seed = 0U;

    // Indices into the mesh shape whose device coordinate should contribute a distinct RNG stream.
    // Axes listed here are treated as data-parallel (each device must draw different noise),
    // axes omitted are treated as replicated (every device must draw the SAME noise or the replicas desync).
    // Empty => every device draws identical noise.
    std::vector<uint32_t> seed_axes{};
};

struct GumbelSampleInputs {
    // Logits, TILE layout, shape [B, 1, tokens, V].
    const ttnn::Tensor& logits;

    // Optional additive mask, subtracted from the scores (after temperature scaling in the sampled
    // path; from the raw logits in the greedy path). [1, 1, 1, V]
    // (one row shared by every batch entry) or [B, 1, 1, V] (one row per entry -- per-request logit
    // bias), always broadcast down token positions. See gumbel_sample.hpp for the unit note.
    std::optional<ttnn::Tensor> logits_mask;

    // OPTIONAL per-batch-entry token position to sample at: [B, 1, 1, 1] UINT32 ROW_MAJOR
    // INTERLEAVED -- byte-for-byte this op's OWN position-mode output spec, so page e of this tensor
    // IS batch entry e IS output page e, one indexing convention end to end.
    //
    // Absent: sample every token position, output [B, 1, tokens, 1].
    //
    // Present: sample ONLY row positions[b] of batch entry b, output [B, 1, 1, 1]. Prefill needs
    // exactly one token per sequence -- the position of that sequence's last real prompt token --
    // but the logits carry all `tokens` positions, so sampling everything does `tokens` times the
    // necessary work and throws away all but one row. Those positions differ per row, so no uniform
    // slice can express them.
    //
    // A device tensor's shape IS its local shard, so there is no global->local mapping for the op to
    // re-derive: the caller shards this with the SAME mapper it sharded the batch with, which makes
    // shard agreement true by construction rather than by two mapper configs happening to match.
    std::optional<ttnn::Tensor> positions;

    std::optional<ttnn::Tensor> preallocated_output;
};

using operation_attributes_t = GumbelSampleParams;
using tensor_args_t = GumbelSampleInputs;

// Sampled token ids, ROW_MAJOR UINT32, shape [B, 1, tokens, 1] -- matching what
// ttnn::argmax(..., dim=3, keepdim=true) produces today so callers need no changes.
using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

}  // namespace ttml::metal::ops::gumbel_sample::device
