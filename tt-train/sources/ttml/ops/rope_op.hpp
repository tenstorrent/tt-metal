// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "autograd/tensor.hpp"

// Forward declaration to avoid heavy include in files that only need the pointer type
namespace ttnn::distributed {
class TensorToMesh;
}  // namespace ttnn::distributed

namespace ttml::ops {

struct RopeScalingParams {
    uint32_t original_context_length = 2048U;
    float scaling_factor = 1.0F;
    float high_freq_factor = 1.0F;
    float low_freq_factor = 1.0F;
};

struct RopeRows;
struct RopeRowsCache;

struct RotaryEmbeddingParams {
    ttnn::Tensor cos_cache{};
    ttnn::Tensor sin_cache{};
    ttnn::Tensor neg_cos_cache{};
    ttnn::Tensor neg_sin_cache{};
    // [sequence_length, head_dim] ROW_MAJOR copies of cos_cache / sin_cache, read by gather_rows.
    ttnn::Tensor cos_table{};
    ttnn::Tensor sin_table{};
    ttnn::Tensor trans_mat{};
    // Per-row cos/sin of the last position_ids passed to rope; shared by copies of these params.
    std::shared_ptr<RopeRowsCache> rows_cache;

    uint32_t sequence_length = 0U;
    uint32_t head_dim = 0U;
    float theta = 10000.0F;

    RopeScalingParams rope_scaling_params;
};

class RotaryEmbeddingPositionIds {
public:
    explicit RotaryEmbeddingPositionIds(autograd::TensorPtr ids);
    void update(autograd::TensorPtr ids);
    [[nodiscard]] const autograd::TensorPtr& get_ids() const;

private:
    friend const RopeRows& get_rope_rows(
        const RotaryEmbeddingParams& params, const RotaryEmbeddingPositionIds& position_ids, uint32_t num_heads);

    static uint64_t s_version_counter;
    autograd::TensorPtr m_ids;
    uint64_t m_version = 0;
};

autograd::TensorPtr rope(
    const autograd::TensorPtr& input, const RotaryEmbeddingParams& rope_params, const uint32_t token_position);

// position_ids: [B, S] UINT32 ROW_MAJOR, the absolute position of every query row.
// Every id must be < rope_params.sequence_length; this is not checked on device.
// cos/sin are reused until position_ids gets a new version; call update() after changing the ids.
autograd::TensorPtr rope(
    const autograd::TensorPtr& input,
    const RotaryEmbeddingParams& rope_params,
    const RotaryEmbeddingPositionIds& position_ids);

std::pair<ttnn::Tensor, ttnn::Tensor> gen_freqs(
    uint32_t head_dim,
    uint32_t sequence_length,
    float theta,
    const RopeScalingParams& rope_scaling_params,
    const ttnn::distributed::TensorToMesh* mesh_mapper = nullptr);

ttnn::Tensor gen_trans_mat();

RotaryEmbeddingParams build_rope_params(
    uint32_t sequence_length,
    uint32_t head_dim,
    float theta = 10000.0F,
    RopeScalingParams rope_scaling_params = RopeScalingParams{});
// Throws an exception if the input is bad, parameters are bad, or the two are
// incompatible with one another.
void validate_rope_input_and_params(const autograd::TensorPtr& input, const RotaryEmbeddingParams& rope_params);

}  // namespace ttml::ops
