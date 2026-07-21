// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::sdpa_fw::device {

struct SDPAForwardParams {
    bool return_intermediates{false};
    AttentionMaskType mask_type{AttentionMaskType::Causal};
    float dropout_probability{0.0F};
};

struct SDPAForwardInputs {
    const ttnn::Tensor& query;
    const ttnn::Tensor& key;
    const ttnn::Tensor& value;
    const std::optional<ttnn::Tensor>& mask;  // attention mask

    std::optional<ttnn::Tensor> preallocated_intermediate;
    std::optional<ttnn::Tensor> preallocated_output;
};

using operation_attributes_t = SDPAForwardParams;
using tensor_args_t = SDPAForwardInputs;

using tensor_return_value_t = std::vector<ttnn::Tensor>;

using spec_return_value_t = std::vector<ttnn::TensorSpec>;

}  // namespace ttml::metal::ops::sdpa_fw::device
