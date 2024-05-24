// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

struct Pad {
    const Shape output_tensor_shape;
    const Shape input_tensor_start;
    const float pad_value;
    const MemoryConfig output_mem_config;
    const bool use_multicore;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "output_tensor_shape", "input_tensor_start", "pad_value", "output_mem_config", "use_multicore");
    const auto attribute_values() const {
        return std::forward_as_tuple(
            this->output_tensor_shape,
            this->input_tensor_start,
            this->pad_value,
            this->output_mem_config,
            this->use_multicore);
    }
};

operation::ProgramWithCallbacks pad_rm_reader_writer(
    const Tensor &a,
    Tensor &output,
    const Shape &output_tensor_shape,
    const Shape &input_tensor_start,
    const float pad_value);
operation::ProgramWithCallbacks pad_rm_reader_writer_multi_core(
    const Tensor &a,
    Tensor &output,
    const Shape &output_tensor_shape,
    const Shape &input_tensor_start,
    const float pad_value);
Tensor pad(
    const Tensor &input_tensor_a,
    const Shape &output_tensor_shape,
    const Shape &input_tensor_start,
    const float pad_value,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    bool use_multicore = false);

struct PadOnHost {
    const Shape output_tensor_shape;
    const Shape input_tensor_start;
    const float pad_value;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> compute_output_tensors(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names =
        std::forward_as_tuple("output_tensor_shape", "input_tensor_start", "pad_value");
    const auto attribute_values() const {
        return std::forward_as_tuple(this->output_tensor_shape, this->input_tensor_start, this->pad_value);
    }
};

Tensor pad_on_host(
    const Tensor &input_tensor_a, const Shape &output_tensor_shape, const Shape &input_tensor_start, float pad_value);

}  // namespace tt_metal

}  // namespace tt
