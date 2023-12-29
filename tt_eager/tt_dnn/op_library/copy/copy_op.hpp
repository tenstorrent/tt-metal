// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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

enum class CopyOpParallelizationStrategy {
    MULTI_CORE = 0, SINGLE_CORE = 1
};

struct Copy {
    const MemoryConfig output_mem_config;
    const DataType output_dtype;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    CopyOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

operation::ProgramWithCallbacks copy_multi_core(const Tensor &input, const Tensor &output, bool backwards = false);
operation::ProgramWithCallbacks copy_single_core(const Tensor &input, const Tensor &output, bool backwards = false);

inline Tensor copy(const Tensor& src_tensor, const Tensor& dst_tensor) {
    operation::run(Copy{dst_tensor.memory_config(), dst_tensor.dtype()}, {src_tensor, dst_tensor});
    return dst_tensor;
}

inline Tensor clone(const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype = std::nullopt) {
    return operation::run(Copy{output_mem_config, output_dtype.value_or(input.dtype())}, {input}).at(0);
}

//unary assign
inline Tensor assign(const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, std::optional<const DataType> output_dtype = std::nullopt) {
    return operation::run(Copy{output_mem_config, output_dtype.value_or(input.dtype())}, {input}).at(0);
}

// binary assign
inline Tensor assign(const Tensor& input_a, const Tensor& input_b) {
    operation::run(Copy{input_b.memory_config(), input_b.dtype()}, {input_a, input_b});
    return input_b;
}

}  // namespace tt_metal

}  // namespace tt
