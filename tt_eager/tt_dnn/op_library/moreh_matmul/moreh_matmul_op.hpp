/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <optional>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

void get_tensor_dim(std::vector<uint32_t> &dim, const Shape &shape);
std::vector<int64_t> find_reduce_dim(const Shape &a_shape, const Shape &b_shape);
bool is_same_batch_dim(const Tensor &tensor_a, const Tensor &tensor_b);

operation::ProgramWithCallbacks moreh_matmul_multi_core(
    const Tensor &input,
    const Tensor &other,
    const Tensor &output,
    const std::optional<const Tensor> &bias,
    bool transpose_input,
    bool transpose_other);

struct MorehMatmul {
    const MemoryConfig output_mem_config;
    bool transpose_input;
    bool transpose_other;
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
    const operation::Hash compute_program_hash(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors) const;
    static constexpr auto attribute_names = std::forward_as_tuple("transpose_input", "transpose_other");
    const auto attribute_values() const { return std::forward_as_tuple(this->transpose_input, this->transpose_other); }
};

Tensor moreh_matmul(
    const Tensor &input,
    const Tensor &other,
    bool transpose_input = false,
    bool transpose_other = false,
    const std::optional<const Tensor> output = std::nullopt,
    const std::optional<const Tensor> bias = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
