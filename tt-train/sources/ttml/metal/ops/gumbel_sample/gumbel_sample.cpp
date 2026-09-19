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
    // The fused op requires the mask to match the logits dtype (one CB format serves both
    // streams), so a mismatched mask is normalized once here, at the op boundary, rather than by
    // every caller. Host tensors are passed through untouched: the device op's validation owns
    // that rejection, and probing dtypes here would just preempt its clearer error.
    // TODO: the follow-up PR in the stack #56181 makes the callsite mask match the logits dtype, so we won't need this
    // bridge anymore then.
    std::optional<ttnn::Tensor> mask = logits_mask;
    if (mask.has_value() && logits.storage_type() == ttnn::StorageType::DEVICE && mask->dtype() != logits.dtype()) {
        mask = ttnn::typecast(*mask, logits.dtype());
    }
    return ttnn::prim::ttml_gumbel_sample(logits, temperature, seed, seed_axes, mask, positions);
}

}  // namespace ttml::metal
