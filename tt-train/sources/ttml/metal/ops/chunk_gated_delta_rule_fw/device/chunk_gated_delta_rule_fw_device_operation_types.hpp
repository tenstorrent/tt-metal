// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::chunk_gated_delta_rule_fw::device {

// ---------------------------------------------------------------------------------------------
// Prep: the state-independent per-(value head, chunk) work, spread over the whole grid.
// ---------------------------------------------------------------------------------------------
struct ChunkGatedDeltaRuleForwardPrepParams {
    uint32_t chunk_size = 64U;
    uint32_t key_head_dim = 0U;  // K; the flat q width is only HK * K, so it cannot be inferred
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    tt::tt_metal::MemoryConfig output_memory_config;
};

// q (pre-scaled), k [B,1,T,HK*K] and v [B,1,T,HV*V] BFLOAT16; g, beta [B,1,T,HV].
struct ChunkGatedDeltaRuleForwardPrepInputs {
    ttnn::Tensor q;
    ttnn::Tensor k;
    ttnn::Tensor v;
    ttnn::Tensor g;
    ttnn::Tensor beta;
};

// All fp32 TILE, BH = B*HV and NC = T/C:
//   kVBeta [BH,NC,C,V]   kKd [BH,NC,C,K]   kQDecay [BH,NC,C,K]   kIntra [BH,NC,C,C]
//   kKDecT [BH,NC,K,C]   kDl [BH,NC,1,1]   kA [BH,1,T,C]         kGCumsum [BH,1,T,1]
enum PrepOutputIndex : size_t { kVBeta = 0, kKd, kQDecay, kIntra, kKDecT, kDl, kA, kGCumsum, kNumPrepOutputs };

// ---------------------------------------------------------------------------------------------
// Scan: the recurrence over chunks, one core per (value head, V-block).
// ---------------------------------------------------------------------------------------------
struct ChunkGatedDeltaRuleForwardScanParams {
    uint32_t num_value_heads = 0U;   // HV; the scan inputs only carry BH = B * HV
    bool output_head_major = false;  // o as [BH,1,T,V] instead of [B,1,T,HV*V]
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    tt::tt_metal::MemoryConfig output_memory_config;
};

// The prep outputs kVBeta..kA, plus the optional fp32 initial_state [B,HV,K,V].
struct ChunkGatedDeltaRuleForwardScanInputs {
    ttnn::Tensor v_beta;
    ttnn::Tensor kd;
    ttnn::Tensor q_decay;
    ttnn::Tensor intra;
    ttnn::Tensor k_dec_t;
    ttnn::Tensor dl;
    ttnn::Tensor a;
    std::optional<ttnn::Tensor> initial_state;
};

// All fp32 TILE:
//   kO [B,1,T,HV*V] (or [BH,1,T,V])   kFinalState [B,HV,K,V]   kH [BH,NC,K,V] (state entering each chunk)   kVNew
//   [BH,1,T,V]
enum ScanOutputIndex : size_t { kO = 0, kFinalState, kH, kVNew, kNumScanOutputs };

using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::chunk_gated_delta_rule_fw::device
