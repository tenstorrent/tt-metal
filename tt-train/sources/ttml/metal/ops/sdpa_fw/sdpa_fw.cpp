// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_fw.hpp"

#include "device/sdpa_fw_device_operation.hpp"

namespace ttml::metal::ops::sdpa_fw {

std::vector<std::optional<ttnn::Tensor>> SDPAForwardOperation::invoke(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const std::optional<ttnn::Tensor>& mask,
    const uint32_t q_heads,
    const uint32_t kv_heads,
    const float dropout_probability,
    const bool return_intermediates) {
    auto result = ttnn::prim::ttml_sdpa_fw(query, key, value, mask, q_heads, kv_heads, dropout_probability, return_intermediates);

    if (result.size() == 1U) {
        return {result[0], std::nullopt};
    }

    return {result[0], result[1]};  // maybe I need to return more than 2 tensors in the future
};

}  // namespace ttml::metal::ops::sdpa_fw
