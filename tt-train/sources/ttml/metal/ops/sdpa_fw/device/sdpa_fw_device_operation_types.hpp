// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ops/sdpa_fw/sdpa_fw_program_config.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::sdpa_fw::device {

struct SDPAForwardParams {
    bool return_intermediates{false};
    AttentionMaskType mask_type{AttentionMaskType::Causal};
    float dropout_probability{0.0F};
    bool has_gate{false};
    // Flattened SDPAForwardProgramConfig (0 = auto). Part of the program-cache hash.
    uint32_t q_chunk_size{0};
    uint32_t k_chunk_size{0};
};

struct SDPAForwardInputs {
    const ttnn::Tensor& query;
    const ttnn::Tensor& key;
    const ttnn::Tensor& value;
    const std::optional<ttnn::Tensor>& mask;  // attention mask

    std::optional<ttnn::Tensor> preallocated_intermediate;
    std::optional<ttnn::Tensor> preallocated_output;

    const std::optional<ttnn::Tensor>& gate;  // optional gate input
};

using operation_attributes_t = SDPAForwardParams;
using tensor_args_t = SDPAForwardInputs;

using tensor_return_value_t = std::vector<ttnn::Tensor>;

using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttml::metal::ops::sdpa_fw::device
