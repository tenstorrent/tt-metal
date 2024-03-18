// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

enum class RepeatOpParallelizationStrategy { SINGLE_CORE = 0, MULTI_CORE = 1 };

struct Repeat {
    const uint32_t repeat_dim;
    const uint32_t num_repeats;
    const MemoryConfig output_mem_config;
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    RepeatOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("repeat_dim", "num_repeats", "output_mem_config");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->repeat_dim), std::cref(this->num_repeats), std::cref(this->output_mem_config));
    }
};

operation::ProgramWithCallbacks repeat_multi_core(
    const Tensor &input_tensor, const uint32_t repeat_dim, const uint32_t num_repeats, const Tensor &output);
operation::ProgramWithCallbacks repeat_single_core(
    const Tensor &input_tensor, const uint32_t repeat_dim, const uint32_t num_repeats, const Tensor &output);

Tensor repeat(
    const Tensor &input_tensor,
    const Shape &shape,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt_metal

}  // namespace tt
