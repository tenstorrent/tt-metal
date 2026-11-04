// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::rmsnorm_fw::device {

struct RMSNormForwardParams {
    bool return_intermediates{false};
    float epsilon{1e-6F};
};

struct RMSNormForwardInputs {
    const ttnn::Tensor& input;
    const ttnn::Tensor& gamma;
    // Gated mode: output = rmsnorm(input) * gamma * silu(gate). `input` is head-major [B*H, 1, T, V] and `gate`
    // head-merged [B, 1, T, H*V]; the output takes the gate's layout.
    std::optional<ttnn::Tensor> gate;

    std::optional<ttnn::Tensor> preallocated_rms;
    std::optional<ttnn::Tensor> preallocated_output;
};

using operation_attributes_t = RMSNormForwardParams;
using tensor_args_t = RMSNormForwardInputs;

using tensor_return_value_t = std::vector<ttnn::Tensor>;

using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::rmsnorm_fw::device
