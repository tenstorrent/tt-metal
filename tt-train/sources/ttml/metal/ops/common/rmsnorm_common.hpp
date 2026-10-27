// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string_view>

#include "metal/common/tensor_validation.hpp"
#include "metal/ttnn_all_includes.hpp"

// Shared by rmsnorm_fw and rmsnorm_bw: the shapes of rms and of the tensors that follow the gate, and the input and
// gamma checks both validates run.
namespace ttml::metal::ops::rmsnorm {

// One rms per row: the input's shape with the last dim 1.
inline ttnn::Shape get_rms_shape(const ttnn::Tensor& input) {
    auto shape = input.logical_shape();
    shape[-1] = 1U;
    return shape;
}

// The forward output, dL_dout and dL_dgate take the gate's head-merged shape in gated mode and the input's otherwise.
inline const ttnn::Shape& get_output_shape(const ttnn::Tensor& input, const std::optional<ttnn::Tensor>& gate) {
    return gate.has_value() ? gate->logical_shape() : input.logical_shape();
}

// A rank-4 input and a gamma of [1, 1, 1, V] for the input's V.
inline void check_input_and_gamma_shapes(const ttnn::Tensor& input, const ttnn::Tensor& gamma, std::string_view op) {
    const auto& input_shape = input.logical_shape();
    TT_FATAL(input_shape.rank() == 4U, "{}: Input must be rank 4, got shape {}", op, input_shape);
    check_shape(gamma, op, "Gamma", ttnn::Shape({1U, 1U, 1U, input_shape[-1]}));
}

}  // namespace ttml::metal::ops::rmsnorm
