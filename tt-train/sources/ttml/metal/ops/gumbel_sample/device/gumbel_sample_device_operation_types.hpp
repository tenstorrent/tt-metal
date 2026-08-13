// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gumbel_sample::device {

struct GumbelSampleParams {
    // Softmax temperature, >= 0. Zero selects greedy argmax: the compute kernel compiles out the
    // RNG and the scaling (DO_GUMBEL_NOISE) and passes the logits straight to the writer's running
    // argmax. Note this makes `temperature > 0` part of the program hash, since it changes the
    // kernel binary -- see compute_program_hash.
    float temperature = 1.0F;

    // RNG seed. Any value is valid, INCLUDING 0 -- it is an ordinary seed here, not a sentinel.
    // This op seeds the SFPU generator directly (rand_tile_init) instead of going through
    // ttnn::rand, so it does not inherit that op's "seed == 0 => host time-based entropy"
    // contract; sampling here is always reproducible for a given seed. Zero is safe because the
    // hardware generator is an XNOR LFSR whose sole lock-up state is all-ones, which
    // ckernel_sfpu_rand.h already rewrites.
    uint32_t seed = 0U;

    // Indices into the mesh shape whose device coordinate should contribute a distinct RNG stream.
    // Mirrors ttnn_fixed::sample's `seed_axes`: axes listed here are treated as data-parallel (each
    // device must draw different noise), axes omitted are treated as replicated (every device must
    // draw the SAME noise or the replicas desync). Empty => every device draws identical noise.
    std::vector<uint32_t> seed_axes{};
};

struct GumbelSampleInputs {
    // Logits, TILE layout, shape [B, 1, tokens, V].
    const ttnn::Tensor& logits;

    // Optional additive padding mask with the same shape as `logits`; subtracted from the scores.
    std::optional<ttnn::Tensor> logits_padding_mask;

    std::optional<ttnn::Tensor> preallocated_output;
};

using operation_attributes_t = GumbelSampleParams;
using tensor_args_t = GumbelSampleInputs;

// Sampled token ids, ROW_MAJOR UINT32, shape [B, 1, tokens, 1] -- matching what
// ttnn::argmax(..., dim=3, keepdim=true) produces today so callers need no changes.
using tensor_return_value_t = ttnn::Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

}  // namespace ttml::metal::ops::gumbel_sample::device
