// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

struct MorehGroupNorm {
    uint32_t num_groups;
    float eps;
    MemoryConfig output_mem_config;
    MemoryConfig mean_mem_config;
    MemoryConfig rstd_mem_config;

    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        const std::vector<std::optional<Tensor>> &output_tensors) const;

    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;

    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;

    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names =
        std::make_tuple("num_groups", "eps", "output_mem_config", "mean_mem_config", "rstd_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->num_groups),
            std::cref(this->eps),
            std::cref(this->output_mem_config),
            std::cref(this->mean_mem_config),
            std::cref(this->rstd_mem_config));
    }
};

operation::ProgramWithCallbacks moreh_groupnorm_impl(
    const Tensor &input,
    uint32_t num_groups,
    float eps,
    Tensor &output,
    Tensor &mean,
    Tensor &rstd,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta);

std::vector<Tensor> moreh_groupnorm(
    const Tensor &input,
    uint32_t num_groups,
    float eps,
    const std::optional<const Tensor> gamma = std::nullopt,
    const std::optional<const Tensor> beta = std::nullopt,
    const std::optional<const Tensor> output = std::nullopt,
    const std::optional<const Tensor> mean = std::nullopt,
    const std::optional<const Tensor> rstd = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const MemoryConfig &mean_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    const MemoryConfig &rstd_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
