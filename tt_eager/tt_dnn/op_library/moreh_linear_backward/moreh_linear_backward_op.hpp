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

// TODO: Move bias backward code
tt_metal::operation::ProgramWithCallbacks moreh_bias_backward_multi_core_h(
    const tt_metal::Tensor &output_grad, const tt_metal::Tensor &bias_grad);

tt_metal::operation::ProgramWithCallbacks moreh_bias_backward_single_core_hw(
    const tt_metal::Tensor &output_grad, const tt_metal::Tensor &bias_grad);

struct MorehBiasBackward {
    void validate(const std::vector<tt_metal::Tensor> &inputs) const;
    std::vector<tt_metal::Shape> compute_output_shapes(const std::vector<tt_metal::Tensor> &inputs) const;
    std::vector<tt_metal::Tensor> create_output_tensors(const std::vector<tt_metal::Tensor> &inputs) const;
    tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<tt_metal::Tensor> &inputs, std::vector<tt_metal::Tensor> &outputs) const;
    tt::stl::reflection::Attributes attributes() const;
};

[[maybe_unused]] std::vector<std::variant<tt_metal::Tensor, char *>> moreh_linear_backward(
    const tt_metal::Tensor &output_grad,
    const tt_metal::Tensor &input,
    const tt_metal::Tensor &weight,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> input_grad = std::nullopt,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> weight_grad = std::nullopt,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> bias_grad = std::nullopt,
    const tt_metal::MemoryConfig &output_mem_config = tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary
}  // namespace operations
}  // namespace tt
