// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::swiglu_fw::device {

// Algorithm selection for SwiGLU
enum class SwiGLUAlgorithm {
    // Original algorithm: materializes full M row in L1
    // Memory: ~560 KB for NanoLlama3
    // Best when: hidden_dim fits comfortably in L1
    ORIGINAL,

    // True Flash algorithm: computes M tiles on-demand
    // Memory: ~280 KB for NanoLlama3 (50% reduction)
    // Trade-off: 8× more X reads (mitigated by X caching in Phase 3)
    // Best when: L1 is constrained or preparing for block matmul
    TRUE_FLASH,

    // Auto-select based on L1 availability and dimensions
    AUTO
};

struct operation_attributes_t {
    SwiGLUAlgorithm algorithm = SwiGLUAlgorithm::AUTO;
};

struct tensor_args_t {
    ttnn::Tensor input;
    ttnn::Tensor w1;
    ttnn::Tensor w2;
    ttnn::Tensor w3;
    std::optional<ttnn::Tensor> preallocated_swiglu = std::nullopt;
};

// Output tensor specs and tensors
using spec_return_value_t = std::vector<ttnn::TensorSpec>;

using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::swiglu_fw::device
