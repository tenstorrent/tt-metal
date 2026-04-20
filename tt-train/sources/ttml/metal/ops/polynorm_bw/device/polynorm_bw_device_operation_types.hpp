// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::polynorm3_bw::device {

struct operation_attributes_t {
    float epsilon{1e-5F};
};

struct tensor_args_t {
    const ttnn::Tensor& input;
    const ttnn::Tensor& dL_dout;
    const ttnn::Tensor& weight;
    std::optional<ttnn::Tensor> preallocated_dL_dx = std::nullopt;
    std::optional<ttnn::Tensor> preallocated_packed_partials = std::nullopt;
};

using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttml::metal::ops::polynorm3_bw::device
