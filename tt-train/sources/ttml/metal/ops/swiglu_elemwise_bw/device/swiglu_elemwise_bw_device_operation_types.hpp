// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::swiglu_elemwise_bw::device {

struct operation_attributes_t {};

struct tensor_args_t {
    ttnn::Tensor linear1;
    ttnn::Tensor gate;
    ttnn::Tensor dL_dprod;
    std::optional<ttnn::Tensor> preallocated_dL_dlinear1 = std::nullopt;
    std::optional<ttnn::Tensor> preallocated_dL_dgate = std::nullopt;
};

struct tensor_return_value_t {
    ttnn::Tensor dL_dlinear1;
    ttnn::Tensor dL_dgate;
};

using spec_return_value_t = std::vector<ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::swiglu_elemwise_bw::device
