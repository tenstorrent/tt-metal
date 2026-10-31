// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::rmsnorm_bw::device {

// Which outputs to produce. dL_dgate is produced whenever there is a gate.
struct RMSNormBackwardOutputs {
    // Without it no dL_da is computed or written, e.g. for a frozen input. Plain mode then also skips gamma and the
    // scale reduction; gated mode still needs gamma for dL_dgate.
    bool da = true;
    // Without it no dL_dgamma_components are computed or written, e.g. for a frozen gamma.
    bool dgamma = true;
};

// Attributes for the backward operation (add more if needed). The backward reads rms from the forward, so it takes
// no epsilon.
struct RMSNormBackwardParams {
    RMSNormBackwardOutputs outputs{};
};

// Tensors required for backward
struct RMSNormBackwardInputs {
    ttnn::Tensor input;
    ttnn::Tensor gamma;
    ttnn::Tensor rms;
    ttnn::Tensor dL_dout;
    // Gated mode (see rmsnorm_fw): dL_dout and dL_dgate are head-merged like the gate, dL_da is head-major.
    std::optional<ttnn::Tensor> gate = std::nullopt;
    std::optional<ttnn::Tensor> preallocated_da = std::nullopt;
    std::optional<ttnn::Tensor> preallocated_dgamma_components = std::nullopt;
    std::optional<ttnn::Tensor> preallocated_dgate = std::nullopt;
};

using operation_attributes_t = RMSNormBackwardParams;
using tensor_args_t = RMSNormBackwardInputs;

// {dL_da, dL_dgamma_components, dL_dgate}; dL_da is nullopt without outputs.da, dL_dgamma_components without
// outputs.dgamma and dL_dgate without a gate.
using spec_return_value_t = std::vector<std::optional<tt::tt_metal::TensorSpec>>;
using tensor_return_value_t = std::vector<std::optional<ttnn::Tensor>>;

}  // namespace ttml::metal::ops::rmsnorm_bw::device
