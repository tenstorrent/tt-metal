// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::polynorm3_bw::device {

struct PolyNorm3BWAttributes {
    float epsilon{1e-5F};
    // Which outputs to compute. A skipped output is nullopt and its kernel work is skipped. The packed partials
    // hold both dw and db, so they are needed when either is.
    bool compute_grad_input{true};
    bool compute_packed_partials{true};
};

struct PolyNorm3BWTensorArgs {
    const ttnn::Tensor& input;
    const ttnn::Tensor& dL_dout;
    const ttnn::Tensor& weight;
    std::optional<ttnn::Tensor> preallocated_dL_dx = std::nullopt;
    std::optional<ttnn::Tensor> preallocated_packed_partials = std::nullopt;
};

// [dL_dx, packed_partials]; a skipped output is nullopt.
using PolyNorm3BWSpecReturn = std::vector<std::optional<tt::tt_metal::TensorSpec>>;
using PolyNorm3BWTensorReturn = std::vector<std::optional<ttnn::Tensor>>;

}  // namespace ttml::metal::ops::polynorm3_bw::device
