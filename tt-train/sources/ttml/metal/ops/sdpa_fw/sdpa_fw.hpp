// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

std::vector<std::optional<ttnn::Tensor>> sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    AttentionMaskType mask_type = AttentionMaskType::Causal,
    const std::optional<ttnn::Tensor>& mask = std::nullopt,  // only used when mask_type == Arbitrary
    const float dropout_probability = 0.0F,
    const bool return_intermediates = false);

}  // namespace ttml::metal
