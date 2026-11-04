// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "autograd/tensor.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

// `seed_axes` (optional): mesh axes across which the logits hold DISTINCT data and must be seeded
// uniquely (e.g. dp / fsdp). Axes omitted are treated as replicated (e.g. tp) and get identical noise.
// std::nullopt (default) => seed NO axis: every device draws the same noise (original behavior).
// `positions` (optional): per-batch-row token position, a [B, 1, 1, 1] UINT32 ROW_MAJOR tensor
// sharded with the SAME mapper as the batch. nullptr (default) samples every position. Supplying it
// samples ONLY those rows and returns [B, 1, 1, 1] -- what prefill actually needs, at a fraction of
// the work, since each sequence's prompt ends at its own position and everything else in the
// [B, 1, tokens, V] logits is thrown away.
autograd::TensorPtr sample_op(
    const autograd::TensorPtr& logits,
    float temperature,
    uint32_t seed,
    const autograd::TensorPtr& logits_mask = nullptr,
    std::optional<std::vector<uint32_t>> seed_axes = std::nullopt,
    const autograd::TensorPtr& positions = nullptr);

}  // namespace ttml::ops
