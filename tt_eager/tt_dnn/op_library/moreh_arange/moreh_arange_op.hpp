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

// The 'any' Tensor arg is only used to pass the device and resulting tensor dtype
struct MorehArange {
    float start;
    float end;
    float step;
    const CoreRange core_range;  // unused for now
    const MemoryConfig output_mem_config;
    const bool inplace;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

Tensor moreh_arange(float start, float end, float step, const Tensor& any, const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor moreh_arange_inplace(Tensor &input_tensor, float start, float end, float step);

}  // namespace primary
}  // namespace operations
}  // namespace tt
