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

using namespace tt_metal;

operation::ProgramWithCallbacks moreh_sum_multi_core(const Tensor &input, const Tensor &output);

struct MorehSum {
    void validate(const std::vector<Tensor> &inputs) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &inputs) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &inputs) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) const;
    stl::reflection::Attributes attributes() const;
};

/*
 * GENERAL matmul_backward
 */
[[maybe_unused]] std::vector<std::variant<Tensor, char *>> moreh_matmul_backward(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &other,
    std::optional<std::reference_wrapper<const Tensor>> input_grad = std::nullopt,
    std::optional<std::reference_wrapper<const Tensor>> other_grad = std::nullopt,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
