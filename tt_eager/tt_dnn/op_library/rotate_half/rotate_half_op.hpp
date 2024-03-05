// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include "third_party/magic_enum/magic_enum.hpp"

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

enum class RotateHalfOpParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1
};

operation::ProgramWithCallbacks rotate_half_single_core(const Tensor &input_tensor, Tensor &output_tensor);

struct RotateHalf {
    const MemoryConfig output_mem_config;

    RotateHalfOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(
        const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors) const;


    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

inline Tensor rotate_half(const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_FATAL(input_tensor.get_legacy_shape()[-1] % (TILE_WIDTH * 2) == 0, "Input X dim must be divisible into tiles");
    Shape pad_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape());
    FormatParams input_format_params = {.pad_shape=pad_shape, .pad_value=0.0, .target_layout=Layout::TILE};
    return operation::run_with_autoformat(RotateHalf{output_mem_config}, {input_tensor}, {input_format_params}, {Layout::TILE}).at(0);
}


}  // namespace tt_metal

}  // namespace tt
