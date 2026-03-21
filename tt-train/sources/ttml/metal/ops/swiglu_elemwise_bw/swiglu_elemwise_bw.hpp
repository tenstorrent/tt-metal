// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

struct SwigluElemwiseBwResult {
    ttnn::Tensor dL_dlinear1;
    ttnn::Tensor dL_dgate;
};

SwigluElemwiseBwResult swiglu_elemwise_bw(
    const ttnn::Tensor& linear1,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& dL_dprod,
    const std::optional<ttnn::Tensor>& preallocated_dL_dlinear1 = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_dL_dgate = std::nullopt);

}  // namespace ttml::metal
