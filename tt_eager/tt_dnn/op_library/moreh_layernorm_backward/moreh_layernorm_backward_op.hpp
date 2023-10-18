// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

struct MorehLayerNormBackwardInputGrad {
    std::vector<uint32_t> normalized_dims;
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
    tt::stl::reflection::Attributes attributes() const;
};

struct MorehLayerNormBackwardGammaBetaGrad {
    std::vector<uint32_t> normalized_dims;
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
    tt::stl::reflection::Attributes attributes() const;
};

operation::ProgramWithCallbacks moreh_layernorm_backward_input_grad_(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    const std::vector<uint32_t> &normalized_dims,
    const std::optional<const Tensor> gamma = std::nullopt,
    const std::optional<const Tensor> input_grad = std::nullopt);

operation::ProgramWithCallbacks moreh_layernorm_backward_gamma_beta_grad_(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    const std::vector<uint32_t> &normalized_dims,
    const std::optional<const Tensor> gamma_grad = std::nullopt,
    const std::optional<const Tensor> beta_grad = std::nullopt);

}  // namespace tt_metal

namespace operations {

using namespace tt_metal;

namespace primary {

[[maybe_unused]] std::variant<Tensor, char *> moreh_layernorm_backward_input_grad(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    std::vector<uint32_t> normalized_dims,
    std::optional<const Tensor> gamma = std::nullopt,
    std::optional<const Tensor> input_grad = std::nullopt,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

[[maybe_unused]] std::vector<std::variant<Tensor, char *>> moreh_layernorm_backward_gamma_beta_grad(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    std::vector<uint32_t> normalized_dims,
    std::optional<const Tensor> gamma_grad = std::nullopt,
    std::optional<const Tensor> beta_grad = std::nullopt,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

[[maybe_unused]] std::vector<std::variant<Tensor, char *>> moreh_layernorm_backward(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mean,
    const Tensor &rstd,
    std::vector<uint32_t> normalized_dims,
    std::optional<const Tensor> gamma = std::nullopt,
    std::optional<const Tensor> input_grad = std::nullopt,
    std::optional<const Tensor> gamma_grad = std::nullopt,
    std::optional<const Tensor> beta_grad = std::nullopt,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
