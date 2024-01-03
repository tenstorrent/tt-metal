// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_dnn/op_library/operation.hpp"

namespace tt {

namespace tt_metal {
enum class ReduceOpMath {
    SUM = 0, MAX = 1, MIN = 2
};

enum class ReduceOpDim {
    H = 0, W = 1, HW = 2
};

enum class ReduceOpParallelizationStrategy {
    MULTI_CORE_H = 0, MULTI_CORE_W = 1, MULTI_CORE_HW = 2, SINGLE_CORE = 3
};

// TODO: Accept parallelization
operation::ProgramWithCallbacks reduce_single_core(const Tensor &input_tensor, Tensor &output_tensor, ReduceOpMath reduce_math, ReduceOpDim reduce_dim, float scaler = 1.0f);
operation::ProgramWithCallbacks reduce_multi_core_h(const Tensor &input_tensor, Tensor &output_tensor, ReduceOpMath reduce_math, ReduceOpDim reduce_dim, float scaler = 1.0f);
operation::ProgramWithCallbacks reduce_multi_core_w(const Tensor &input_tensor, Tensor &output_tensor, ReduceOpMath reduce_math, ReduceOpDim reduce_dim, float scaler = 1.0f);

struct Reduce {
    const ReduceOpMath math_op;
    const ReduceOpDim dim;
    const float scaler;
    const MemoryConfig output_mem_config;
    const DataType output_dtype;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    ReduceOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("math_op", "dim", "scaler", "output_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->math_op), std::cref(this->dim), std::cref(this->scaler), std::cref(this->output_mem_config), std::cref(this->output_dtype));
    }
};

Tensor reduce(const Tensor &input_tensor, ReduceOpMath reduce_math, ReduceOpDim reduce_dim, float scaler = 1.0f, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, const std::optional<DataType>& output_dtype=std::nullopt);
Tensor max(const Tensor &input_tensor, uint dim, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor sum(const Tensor &input_tensor, uint dim, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor mean(const Tensor& input_tensor, uint aggregate_dims=2, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor mean_hw(const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor global_mean(const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor global_sum(Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
Tensor global_max(Tensor& val, const MemoryConfig& output_mem_config);
Tensor global_min(Tensor& val, const MemoryConfig& output_mem_config);

}  // namespace tt_metal

}  // namespace tt

namespace reduce_op_utils {

using namespace tt::tt_metal;

string dim_to_kernel_name(ReduceOpDim reduce_dim, ReduceOpMath reduce_op);

std::map<string, string> get_defines(ReduceOpMath reduce_op, ReduceOpDim reduce_dim);

} // namespace reduce_op_utils
