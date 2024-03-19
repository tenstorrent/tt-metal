/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/operation.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

// TODO: Move bias backward code
operation::ProgramWithCallbacks moreh_bias_backward_multi_core_h(const Tensor &output_grad, const Tensor &bias_grad);

operation::ProgramWithCallbacks moreh_bias_backward_single_core_hw(const Tensor &output_grad, const Tensor &bias_grad);

struct MorehBiasBackward {
    void validate(const std::vector<Tensor> &inputs) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &inputs) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &inputs) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) const;
    stl::reflection::Attributes attributes() const;
};

[[maybe_unused]] std::vector<std::variant<Tensor, char *>> moreh_linear_backward(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &weight,
    std::optional<const Tensor> input_grad = std::nullopt,
    std::optional<const Tensor> weight_grad = std::nullopt,
    std::optional<const Tensor> bias_grad = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary
}  // namespace operations
}  // namespace tt
