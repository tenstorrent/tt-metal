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

struct MorehLayerNormBackwardInputGrad {
    uint32_t normalized_dims;
    tt_metal::MemoryConfig output_mem_config;

    void validate(
        const std::vector<tt_metal::Tensor> &input_tensors,
        const std::vector<std::optional<const tt_metal::Tensor>> &optional_input_tensors) const;
    std::vector<tt_metal::Shape> compute_output_shapes(const std::vector<tt_metal::Tensor> &input_tensors) const;
    std::vector<tt_metal::Tensor> create_output_tensors(const std::vector<tt_metal::Tensor> &input_tensors) const;
    tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<tt_metal::Tensor> &input_tensors,
        const std::vector<std::optional<const tt_metal::Tensor>> &optional_input_tensors,
        std::vector<tt_metal::Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

struct MorehLayerNormBackwardGammaBetaGrad {
    uint32_t normalized_dims;
    tt_metal::MemoryConfig output_mem_config;

    void validate(
        const std::vector<tt_metal::Tensor> &input_tensors,
        const std::vector<std::optional<const tt_metal::Tensor>> &optional_input_tensors) const;
    std::vector<tt_metal::Shape> compute_output_shapes(const std::vector<tt_metal::Tensor> &input_tensors) const;
    std::vector<tt_metal::Tensor> create_output_tensors(const std::vector<tt_metal::Tensor> &input_tensors) const;
    tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<tt_metal::Tensor> &input_tensors,
        const std::vector<std::optional<const tt_metal::Tensor>> &optional_input_tensors,
        std::vector<tt_metal::Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

tt_metal::operation::ProgramWithCallbacks moreh_layernorm_backward_input_grad_impl(
    const tt_metal::Tensor &output_grad,
    const tt_metal::Tensor &input,
    const tt_metal::Tensor &mean,
    const tt_metal::Tensor &rstd,
    uint32_t normalized_dims,
    const tt_metal::Tensor &input_grad,
    const std::optional<std::reference_wrapper<const tt_metal::Tensor>> gamma = std::nullopt);

tt_metal::operation::ProgramWithCallbacks moreh_layernorm_backward_gamma_beta_grad_impl(
    const tt_metal::Tensor &output_grad,
    const tt_metal::Tensor &input,
    const tt_metal::Tensor &mean,
    const tt_metal::Tensor &rstd,
    uint32_t normalized_dims,
    const std::optional<std::reference_wrapper<const tt_metal::Tensor>> gamma_grad = std::nullopt,
    const std::optional<std::reference_wrapper<const tt_metal::Tensor>> beta_grad = std::nullopt);

[[maybe_unused]] tt_metal::Tensor moreh_layernorm_backward_input_grad(
    const tt_metal::Tensor &output_grad,
    const tt_metal::Tensor &input,
    const tt_metal::Tensor &mean,
    const tt_metal::Tensor &rstd,
    uint32_t normalized_dims,
    const tt_metal::Tensor &input_grad,
    const std::optional<std::reference_wrapper<const tt_metal::Tensor>> gamma = std::nullopt,
    const tt_metal::MemoryConfig &output_mem_config = tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

[[maybe_unused]] std::vector<std::variant<tt_metal::Tensor, char *>> moreh_layernorm_backward_gamma_beta_grad(
    const tt_metal::Tensor &output_grad,
    const tt_metal::Tensor &input,
    const tt_metal::Tensor &mean,
    const tt_metal::Tensor &rstd,
    uint32_t normalized_dims,
    const std::optional<std::reference_wrapper<const tt_metal::Tensor>> gamma_grad = std::nullopt,
    const std::optional<std::reference_wrapper<const tt_metal::Tensor>> beta_grad = std::nullopt,
    const tt_metal::MemoryConfig &output_mem_config = tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

[[maybe_unused]] std::vector<std::variant<tt_metal::Tensor, char *>> moreh_layernorm_backward(
    const tt_metal::Tensor &output_grad,
    const tt_metal::Tensor &input,
    const tt_metal::Tensor &mean,
    const tt_metal::Tensor &rstd,
    uint32_t normalized_dims,
    const std::optional<std::reference_wrapper<const tt_metal::Tensor>> gamma = std::nullopt,
    const std::optional<std::reference_wrapper<const tt_metal::Tensor>> input_grad = std::nullopt,
    const std::optional<std::reference_wrapper<const tt_metal::Tensor>> gamma_grad = std::nullopt,
    const std::optional<std::reference_wrapper<const tt_metal::Tensor>> beta_grad = std::nullopt,
    const tt_metal::MemoryConfig &output_mem_config = tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
