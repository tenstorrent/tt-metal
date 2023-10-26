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

struct MorehLayerNorm {
    float eps;
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

inline Tensor moreh_layernorm(
    const Tensor &input,
    float eps,
    std::vector<uint32_t> normalized_dims,
    std::optional<const Tensor> gamma = std::nullopt,
    std::optional<const Tensor> beta = std::nullopt,
    std::optional<const Tensor> mean = std::nullopt,
    std::optional<const Tensor> rstd = std::nullopt,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return operation::run_with_autoformat(
               MorehLayerNorm{.eps = eps, .normalized_dims = normalized_dims, .output_mem_config = mem_config},
               {input},
               {gamma, beta, mean, rstd})
        .at(0);
}

}  // namespace tt_metal

namespace operations {

using namespace tt_metal;

namespace primary {
inline Tensor moreh_layernorm(
    const Tensor &input,
    float eps,
    std::vector<uint32_t> normalized_dims,
    std::optional<const Tensor> gamma = std::nullopt,
    std::optional<const Tensor> beta = std::nullopt,
    std::optional<const Tensor> mean = std::nullopt,
    std::optional<const Tensor> rstd = std::nullopt,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    return operation::run(
               MorehLayerNorm{.eps = eps, .normalized_dims = normalized_dims, .output_mem_config = mem_config},
               {input},
               {gamma, beta, mean, rstd})
        .at(0);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
