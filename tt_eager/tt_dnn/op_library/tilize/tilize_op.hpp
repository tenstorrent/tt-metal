// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

enum class TilizeOpParallelizationStrategy { MULTI_CORE, SINGLE_CORE };

struct Tilize {
    const MemoryConfig output_mem_config;
    const DataType output_dtype;
    const bool use_multicore;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    TilizeOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
};

enum class TilizeWithValPaddingOpParallelizationStrategy { MULTI_CORE, SINGLE_CORE };

struct TilizeWithValPadding {
    const Shape output_tensor_shape;
    const float pad_value;
    const MemoryConfig output_mem_config;
    const DataType output_dtype;
    const bool use_multicore;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    TilizeWithValPaddingOpParallelizationStrategy get_parallelization_strategy(
        const std::vector<Tensor> &input_tensors) const;
};

operation::ProgramWithCallbacks tilize_multi_core(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks tilize_single_core(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks tilize_with_val_padding_multi_core(
    const Tensor &a, Tensor &output, const float pad_value);

operation::ProgramWithCallbacks tilize_with_val_padding_single_core(
    const Tensor &a, Tensor &output, const float pad_value);

Tensor tilize(
    const Tensor &a,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt,
    bool use_multicore = false);
Tensor tilize_with_zero_padding(
    const Tensor &a,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt,
    bool use_multicore = false);
Tensor tilize_with_val_padding(
    const Tensor &a,
    const Shape &output_tensor_shape,
    const float pad_value,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt,
    bool use_multicore = false);

}  // namespace tt_metal

}  // namespace tt
