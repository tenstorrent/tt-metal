// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::polynorm3_bw::device {

struct PolyNorm3BWAttributes {
    float epsilon{1e-5F};
};

struct PolyNorm3BWTensorArgs {
    const ttnn::Tensor& input;
    const ttnn::Tensor& dL_dout;
    const ttnn::Tensor& weight;
    std::optional<ttnn::Tensor> preallocated_dL_dx = std::nullopt;
    std::optional<ttnn::Tensor> preallocated_packed_partials = std::nullopt;
};

using PolyNorm3BWSpecReturn = std::vector<ttnn::TensorSpec>;
using PolyNorm3BWTensorReturn = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::polynorm3_bw::device
