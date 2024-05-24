// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

struct MorehLayerNorm {
    uint32_t normalized_dims;
    float eps;
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

    static constexpr auto attribute_names = std::forward_as_tuple("normalized_dims", "eps", "output_mem_config");
    const auto attribute_values() const {
        return std::forward_as_tuple(this->normalized_dims, this->eps, this->output_mem_config);
    }
};

operation::ProgramWithCallbacks moreh_layernorm_impl(
    const Tensor &input,
    uint32_t normalized_dims,
    float eps,
    Tensor &output,
    const std::optional<std::reference_wrapper<const Tensor>> gamma,
    const std::optional<std::reference_wrapper<const Tensor>> beta,
    const std::optional<std::reference_wrapper<const Tensor>> mean,
    const std::optional<std::reference_wrapper<const Tensor>> rstd);

Tensor moreh_layernorm(
    const Tensor &input,
    uint32_t normalized_dims,
    float eps,
    const std::optional<std::reference_wrapper<const Tensor>> gamma = std::nullopt,
    const std::optional<std::reference_wrapper<const Tensor>> beta = std::nullopt,
    const std::optional<std::reference_wrapper<const Tensor>> mean = std::nullopt,
    const std::optional<std::reference_wrapper<const Tensor>> rstd = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

namespace tt_metal {

Tensor moreh_layernorm(
    const Tensor &input,
    uint32_t normalized_dims,
    float eps,
    const std::optional<std::reference_wrapper<const Tensor>> gamma = std::nullopt,
    const std::optional<std::reference_wrapper<const Tensor>> beta = std::nullopt,
    const std::optional<std::reference_wrapper<const Tensor>> mean = std::nullopt,
    const std::optional<std::reference_wrapper<const Tensor>> rstd = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt_metal

}  // namespace tt
