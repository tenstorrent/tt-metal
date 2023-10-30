/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace operations {

namespace primary {

/*
 * dot product backward
 */
tt_metal::operation::ProgramWithCallbacks moreh_dot_backward_single_core(
    const tt_metal::Tensor &output_grad,
    const tt_metal::Tensor &input,
    const tt_metal::Tensor &other,
    const std::optional<const tt_metal::Tensor> &input_grad,
    const std::optional<const tt_metal::Tensor> &other_grad);

struct MorehDotBackward {
    void validate(
        const std::vector<tt_metal::Tensor> &inputs,
        const std::vector<std::optional<const tt_metal::Tensor>> &optional_inputs) const;
    std::vector<tt_metal::Shape> compute_output_shapes(const std::vector<tt_metal::Tensor> &inputs) const;
    std::vector<tt_metal::Tensor> create_output_tensors(const std::vector<tt_metal::Tensor> &inputs) const;
    tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<tt_metal::Tensor> &inputs,
        const std::vector<std::optional<const tt_metal::Tensor>> &optional_inputs,
        std::vector<tt_metal::Tensor> &outputs) const;
    tt::stl::reflection::Attributes attributes() const;
};

[[maybe_unused]] std::vector<std::variant<tt_metal::Tensor, char *>> moreh_dot_backward(
    const tt_metal::Tensor &output_grad,
    const tt_metal::Tensor &input,
    const tt_metal::Tensor &other,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> input_grad = std::nullopt,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> other_grad = std::nullopt,
    const tt_metal::MemoryConfig &mem_config = tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
