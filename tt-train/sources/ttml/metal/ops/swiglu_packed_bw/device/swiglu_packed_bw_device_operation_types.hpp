// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::swiglu_packed_bw::device {

struct SwigluPackedBwParams {};

// Backward of swiglu_packed_fw. `packed` [.., R, 2*I] = [gate | up], `dL_dh` [.., R, I];
// produces dL_dpacked [.., R, 2*I] = [dgate | dup].
struct SwigluPackedBwInputs {
    ttnn::Tensor packed;
    ttnn::Tensor dL_dh;
    std::optional<ttnn::Tensor> preallocated_dL_dpacked = std::nullopt;
};

using operation_attributes_t = SwigluPackedBwParams;
using tensor_args_t = SwigluPackedBwInputs;

using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttml::metal::ops::swiglu_packed_bw::device
