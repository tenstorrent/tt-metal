// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::swiglu_elemwise_bw::device {

struct SwigluElemwiseBwParams {};

struct SwigluElemwiseBwInputs {
    ttnn::Tensor linear1;
    ttnn::Tensor gate;
    ttnn::Tensor dL_dprod;
    std::optional<ttnn::Tensor> preallocated_dL_dlinear1 = std::nullopt;
    std::optional<ttnn::Tensor> preallocated_dL_dgate = std::nullopt;
};

struct SwigluElemwiseBwResult {
    ttnn::Tensor dL_dlinear1;
    ttnn::Tensor dL_dgate;
};

using SwigluElemwiseBwOutputSpecs = std::vector<ttnn::TensorSpec>;

// Backward-compat aliases for in-flight branches.
using operation_attributes_t = SwigluElemwiseBwParams;
using tensor_args_t = SwigluElemwiseBwInputs;
using tensor_return_value_t = SwigluElemwiseBwResult;
using spec_return_value_t = SwigluElemwiseBwOutputSpecs;

}  // namespace ttml::metal::ops::swiglu_elemwise_bw::device
