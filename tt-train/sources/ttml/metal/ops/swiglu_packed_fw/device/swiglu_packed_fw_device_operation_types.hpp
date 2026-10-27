// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::swiglu_packed_fw::device {

struct SwigluPackedFwParams {};

// `packed` [.., R, 2*I] = [gate | up]; output h = silu(gate) * up, [.., R, I].
struct SwigluPackedFwInputs {
    ttnn::Tensor packed;
    std::optional<ttnn::Tensor> preallocated_output = std::nullopt;
};

using operation_attributes_t = SwigluPackedFwParams;
using tensor_args_t = SwigluPackedFwInputs;

using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::swiglu_packed_fw::device
