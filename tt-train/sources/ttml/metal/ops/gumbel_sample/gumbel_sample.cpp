// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gumbel_sample.hpp"

#include "device/gumbel_sample_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor gumbel_sample(
    const ttnn::Tensor& logits,
    float temperature,
    uint32_t seed,
    const std::vector<uint32_t>& seed_axes,
    const std::optional<ttnn::Tensor>& logits_mask,
    const std::optional<ttnn::Tensor>& positions) {
    return ttnn::prim::ttml_gumbel_sample(logits, temperature, seed, seed_axes, logits_mask, positions);
}

}  // namespace ttml::metal
