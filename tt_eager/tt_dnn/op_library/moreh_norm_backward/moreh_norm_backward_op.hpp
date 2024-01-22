// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

operation::ProgramWithCallbacks moreh_norm_backward_(
    const Tensor &input, const Tensor &output, const Tensor &output_grad, float p, const Tensor &input_grad);

struct MorehNormBackward {
    float p;
    MemoryConfig input_grad_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("p", "input_grad_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->p), std::cref(this->input_grad_mem_config));
    }
};

[[maybe_unused]] Tensor moreh_norm_backward(
    const Tensor &input,
    const Tensor &output,
    const Tensor &output_grad,
    float p,
    const std::optional<std::reference_wrapper<const Tensor>> input_grad = std::nullopt,
    const MemoryConfig &input_grad_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor moreh_norm_backward_impl(
    const Tensor &input, const Tensor &output, const Tensor &output_grad, float p, const Tensor &input_grad);

}  // namespace primary

}  // namespace operations

}  // namespace tt
