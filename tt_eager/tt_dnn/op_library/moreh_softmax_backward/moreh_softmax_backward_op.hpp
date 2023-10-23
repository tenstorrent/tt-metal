/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tt_dnn/op_library/operation.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

enum class MorehSoftmaxBackwardOpParallelizationStrategy {
    NONE = 0,
    SMALL_W = 1,
    SMALL_H = 2,
    LARGE_W = 3,
    LARGE_H = 4,
    LARGE_C = 5
};

enum class MorehSoftmaxBackwardOp { SOFTMAX = 0, SOFTMIN = 1 };

bool is_moreh_softmax_backward_w_small_available(const Tensor &tensor);
bool is_moreh_softmax_backward_h_small_available(const Tensor &tensor);

operation::ProgramWithCallbacks moreh_softmax_backward_w_small(
    const Tensor &output,
    const Tensor &output_grad,
    Tensor &input_grad,
    const CoreRange core_range,
    const MorehSoftmaxBackwardOp op);
operation::ProgramWithCallbacks moreh_softmax_backward_w_large(
    const Tensor &output,
    const Tensor &output_grad,
    Tensor &input_grad,
    const CoreRange core_range,
    const MorehSoftmaxBackwardOp op);
operation::ProgramWithCallbacks moreh_softmax_backward_h_small(
    const Tensor &output,
    const Tensor &output_grad,
    Tensor &input_grad,
    const CoreRange core_range,
    const MorehSoftmaxBackwardOp op);
operation::ProgramWithCallbacks moreh_softmax_backward_h_large(
    const Tensor &output,
    const Tensor &output_grad,
    Tensor &input_grad,
    const CoreRange core_range,
    const MorehSoftmaxBackwardOp op);
operation::ProgramWithCallbacks moreh_softmax_backward_c_large(
    const Tensor &output,
    const Tensor &output_grad,
    Tensor &input_grad,
    uint32_t dim,
    const CoreRange core_range,
    const MorehSoftmaxBackwardOp op);

struct MorehSoftmaxBackward {
    const uint32_t dim;
    const MemoryConfig output_mem_config;
    const CoreRange core_range;  // unused for now
    const MorehSoftmaxBackwardOp op;
    const MorehSoftmaxBackwardOpParallelizationStrategy strategy;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    MorehSoftmaxBackwardOpParallelizationStrategy get_parallelization_strategy(
        const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

// const ref prevents
Tensor moreh_softmax_backward(
    const Tensor &output_tensor,
    const Tensor &output_grad_tensor,
    uint32_t dim,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor moreh_softmin_backward(
    const Tensor &output_tensor,
    const Tensor &output_grad_tensor,
    uint32_t dim,
    const MorehSoftmaxBackwardOpParallelizationStrategy strategy = MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary
}  // namespace operations
}  // namespace tt
