// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::chunk_gated_delta_rule_bw::device {

struct ChunkGatedDeltaRuleBackwardParams {
    uint32_t chunk_size = 64U;
    uint32_t key_head_dim = 0U;  // K; the flat q width is only HK * K, so it cannot be inferred
    float scale = 0.0F;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    tt::tt_metal::MemoryConfig output_memory_config;
};

// q, k [B,T,HK*K]; v [B,T,HV*V]; g, beta [B,T,HV]; each may also be rank 4 as [B,1,T,X].
// d_out is token-major [B,T,HV*V] or head-major [B*HV,1,T,V] (also rank 3 as [B*HV,T,V]).
// initial_state, d_final_state [B,HV,K,V].
// h [B*HV,NC,K,V], v_new [B*HV,1,T,V], a [B*HV,1,T,C]: the FLOAT32 intermediates of
// ttml::metal::chunk_gated_delta_rule_fw, given all together or not at all.
struct ChunkGatedDeltaRuleBackwardInputs {
    ttnn::Tensor q;
    ttnn::Tensor k;
    ttnn::Tensor v;
    ttnn::Tensor g;
    ttnn::Tensor beta;
    ttnn::Tensor d_out;
    std::optional<ttnn::Tensor> initial_state;
    std::optional<ttnn::Tensor> d_final_state;
    std::optional<ttnn::Tensor> h;
    std::optional<ttnn::Tensor> v_new;
    std::optional<ttnn::Tensor> a;

    [[nodiscard]] bool has_saved_forward() const {
        return h.has_value();
    }
};

// [N,T,X] or [N,1,T,X] -> [N,1,T,X]; any other shape is returned unchanged for the caller to reject.
// A TILE tensor pages both ranks identically, so the kernels never see the difference.
inline ttnn::Shape as_rank4(const ttnn::Shape& shape) {
    if (shape.rank() == 3) {
        return ttnn::Shape({shape[0], 1U, shape[1], shape[2]});
    }
    return shape;
}

// d_out is head-major [B*HV,1,T,V] rather than token-major [B,1,T,HV*V].  With HV == 1 the two
// layouts are the same pages, so that case reads as token-major.
inline bool d_out_is_head_major(const ChunkGatedDeltaRuleBackwardInputs& tensor_args) {
    const auto vs = as_rank4(tensor_args.v.logical_shape());
    const auto ds = as_rank4(tensor_args.d_out.logical_shape());
    return ds != vs && vs.rank() == 4 && ds.rank() == 4;
}

using operation_attributes_t = ChunkGatedDeltaRuleBackwardParams;
using tensor_args_t = ChunkGatedDeltaRuleBackwardInputs;

// Output order: kDq, kDk, kDv, kDg, kDbeta, kScratch, then kDh0 iff initial_state is given.
enum OutputIndex : size_t { kDq = 0, kDk, kDv, kDg, kDbeta, kScratch, kDh0 };

using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::chunk_gated_delta_rule_bw::device
