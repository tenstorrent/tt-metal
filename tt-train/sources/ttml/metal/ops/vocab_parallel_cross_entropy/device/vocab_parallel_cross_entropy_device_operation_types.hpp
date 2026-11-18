// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::vocab_parallel_cross_entropy::device {

// Stats: per shard and row, the shard's record. Combine: the gathered records into the loss and row state.
// Backward: the input gradient from the row state, locally on every shard.
enum class Mode : uint8_t { Stats, Combine, Backward };

struct VocabParallelCrossEntropyParams {
    Mode mode = Mode::Stats;
    // Mesh axis the vocabulary is sharded on; the device's linear mesh index is its shard when nullopt.
    std::optional<uint32_t> cluster_axis;
    float scaler = 1.F;  // Backward only.
};

struct VocabParallelCrossEntropyInputs {
    ttnn::Tensor input;  // Logits (stats, backward) or the gathered shard records (combine).
    std::optional<ttnn::Tensor> target;
    std::optional<ttnn::Tensor> state;
    // Backward: upstream gradient folded into the row scale, one value ([1, 1, 1, 1]) or one per row.
    std::optional<ttnn::Tensor> upstream;
};

// 0: no upstream gradient, 1: one value for every row, 2: one value per row.
inline uint32_t upstream_kind(const VocabParallelCrossEntropyInputs& args) {
    if (!args.upstream.has_value()) {
        return 0;
    }
    return args.upstream->logical_volume() == 1 ? 1 : 2;
}

using operation_attributes_t = VocabParallelCrossEntropyParams;
using tensor_args_t = VocabParallelCrossEntropyInputs;

// Stats: {records}. Combine: {loss, row state}. Backward: {input gradient}.
using tensor_return_value_t = std::vector<ttnn::Tensor>;
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::vocab_parallel_cross_entropy::device
