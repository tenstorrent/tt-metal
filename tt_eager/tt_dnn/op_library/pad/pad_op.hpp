// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

struct PadOnHost {
    const Shape output_tensor_shape;
    const Shape input_tensor_start;
    const float pad_value;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> compute_output_tensors(const std::vector<Tensor> &input_tensors) const;
};

Tensor pad_on_host(const Tensor &input_tensor_a, const Shape &output_tensor_shape, const Shape &input_tensor_start, float pad_value);

}  // namespace tt_metal

}  // namespace tt
