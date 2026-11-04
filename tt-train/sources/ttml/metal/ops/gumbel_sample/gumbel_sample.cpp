// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gumbel_sample.hpp"

#include "device/gumbel_sample_device_operation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

namespace ttml::metal {

ttnn::Tensor gumbel_sample(
    const ttnn::Tensor& logits,
    float temperature,
    uint32_t seed,
    const std::vector<uint32_t>& seed_axes,
    const std::optional<ttnn::Tensor>& logits_mask,
    const std::optional<ttnn::Tensor>& positions) {
    // The fused op requires the mask to match the logits dtype; normalize a mismatch once here, at
    // the op boundary. Host tensors pass through untouched (validation owns that rejection).
    // TODO(#56181): the follow-up PR makes callers build a matching mask; drop this bridge then.
    std::optional<ttnn::Tensor> mask = logits_mask;
    if (mask.has_value() && logits.storage_type() == ttnn::StorageType::DEVICE && mask->dtype() != logits.dtype()) {
        mask = ttnn::typecast(*mask, logits.dtype());
    }
    return ttnn::prim::ttml_gumbel_sample(logits, temperature, seed, seed_axes, mask, positions);
}

}  // namespace ttml::metal
