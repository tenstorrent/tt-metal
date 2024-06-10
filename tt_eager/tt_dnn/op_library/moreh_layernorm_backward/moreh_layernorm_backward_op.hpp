// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <variant>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

struct MorehLayerNormBackwardInputGrad {
    uint32_t normalized_dims;
    MemoryConfig output_mem_config;

    void validate(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
    static constexpr auto attribute_names = std::make_tuple("normalized_dims", "output_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->normalized_dims), std::cref(this->output_mem_config));
    }
};

struct MorehLayerNormBackwardGammaBetaGrad {
    uint32_t normalized_dims;
    MemoryConfig output_mem_config;

    void validate(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
    static constexpr auto attribute_names = std::make_tuple("normalized_dims", "output_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->normalized_dims), std::cref(this->output_mem_config));
    }
};

operation::ProgramWithCallbacks moreh_layernorm_backward_input_grad_impl(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t normalized_dims,
    const Tensor &input_grad,
    const std::optional<const Tensor> gamma = std::nullopt);

operation::ProgramWithCallbacks moreh_layernorm_backward_gamma_beta_grad_impl(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t normalized_dims,
    const std::optional<const Tensor> gamma_grad = std::nullopt,
    const std::optional<const Tensor> beta_grad = std::nullopt);

Tensor moreh_layernorm_backward_input_grad(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t normalized_dims,
    const Tensor &input_grad,
    const std::optional<const Tensor> gamma = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<std::optional<Tensor>> moreh_layernorm_backward_gamma_beta_grad(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t normalized_dims,
    const std::optional<const Tensor> gamma_grad = std::nullopt,
    const std::optional<const Tensor> beta_grad = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

std::vector<std::optional<Tensor>> moreh_layernorm_backward(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    uint32_t normalized_dims,
    const std::optional<const Tensor> gamma = std::nullopt,
    const std::optional<const Tensor> input_grad = std::nullopt,
    const std::optional<const Tensor> gamma_grad = std::nullopt,
    const std::optional<const Tensor> beta_grad = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
