/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

enum class CloneOpParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1
};

struct Clone {
    const MemoryConfig output_mem_config;
    const CloneOpParallelizationStrategy move_op_parallelization_strategy;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    CloneOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

operation::ProgramWithCallbacks clone_multi_core(const Tensor &input, Tensor &output);
operation::ProgramWithCallbacks clone_single_core(const Tensor &input, Tensor &output);

inline Tensor clone(const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    Shape pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.shape());
    FormatParams input_format_params = {.pad_shape=pad_shape, .pad_value=0.0, .target_layout=Layout::TILE};
    return operation::run_with_autoformat(Clone{output_mem_config}, {input_tensor}, {input_format_params}, {Layout::TILE}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
