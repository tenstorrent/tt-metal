/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tt_eager/tensor/tensor.hpp"

#include "tt_dnn/op_library/operation.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

enum class MorehSoftmaxParallelizationStrategy {
    SMALL_W = 0,
    SMALL_H = 1,
    LARGE_W = 2,
    LARGE_H = 3
};

bool is_moreh_softmax_w_small_available(const Tensor &tensor);
bool is_moreh_softmax_h_small_available(const Tensor &tensor);

operation::ProgramWithCallbacks moreh_softmax_w_small(const Tensor &input, Tensor &output, const CoreRange core_range);
operation::ProgramWithCallbacks moreh_softmax_w_large(const Tensor &input, Tensor &output, const CoreRange core_range);
operation::ProgramWithCallbacks moreh_softmax_h_small(const Tensor &input, Tensor &output, const CoreRange core_range);
operation::ProgramWithCallbacks moreh_softmax_h_large(const Tensor &input, Tensor &output, const CoreRange core_range);

struct MorehSoftmax {
    const uint32_t dim;
    const MemoryConfig output_mem_config;
    const CoreRange core_range; // unused for now

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    MorehSoftmaxParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

// const ref prevents
Tensor moreh_softmax(const Tensor& input_tensor, uint32_t dim, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary
}  // namespace operations
}  // namespace tt
