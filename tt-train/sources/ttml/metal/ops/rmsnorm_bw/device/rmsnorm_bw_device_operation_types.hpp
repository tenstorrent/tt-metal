// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::rmsnorm_bw::device {

enum class RMSNormBackwardMode : uint8_t {
    Full,           // dx and dgamma, row reduced locally
    LocalSum,       // only per-row sum(x * gamma * dL_dout) of a row sharded across devices, no rms needed
    ExternalScale,  // dx and dgamma, row reduction taken from gathered LocalSum outputs
};

// Attributes for the backward operation
struct RMSNormBackwardParams {
    float epsilon = 1e-6F;
    RMSNormBackwardMode mode = RMSNormBackwardMode::Full;
    uint32_t full_width = 0U;  // unsharded row width, required by ExternalScale and ignored by the other modes
};

// Tensors required for backward
struct RMSNormBackwardInputs {
    ttnn::Tensor input;
    ttnn::Tensor gamma;
    std::optional<ttnn::Tensor> rms;  // unused by LocalSum
    ttnn::Tensor dL_dout;
    std::optional<ttnn::Tensor> external_sums = std::nullopt;  // [B, 1, S, 32 * num_shards] fp32 (ExternalScale only)
    std::optional<ttnn::Tensor> preallocated_da = std::nullopt;
    std::optional<ttnn::Tensor> preallocated_dgamma_components = std::nullopt;
};

using operation_attributes_t = RMSNormBackwardParams;
using tensor_args_t = RMSNormBackwardInputs;

// Output tensor specs and tensors
// Full / ExternalScale : {dL_da, dgamma}
//   dgamma is bf16 [B, N, S, C] per-row components or fp32 [1, 1, num_cores * 32, C] per-core partials,
//   summed over dims {0, 1, 2} by the caller
// LocalSum             : {sums}
//   sums is fp32 [B, N, S, 32], row sum in column 0, zeros elsewhere
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::rmsnorm_bw::device
