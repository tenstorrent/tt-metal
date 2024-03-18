// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/repeat/repeat_op.hpp"

#include "tensor/tensor_utils.hpp"
#include "tt_dnn/op_library/auto_format.hpp"
#include "tt_dnn/op_library/copy/copy_op.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

RepeatOpParallelizationStrategy Repeat::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    uint32_t num_pages = tt_metal::compute_volume(this->compute_output_shapes(input_tensors).at(0));
    if (input_tensors[0].get_layout() == Layout::ROW_MAJOR) {
        num_pages /= input_tensors[0].get_legacy_shape()[-1];
    } else {
        num_pages /= TILE_HW;
    }
    if (num_pages > 1) {
        return RepeatOpParallelizationStrategy::MULTI_CORE;
    } else {
        return RepeatOpParallelizationStrategy::SINGLE_CORE;
    }
}

void Repeat::validate(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor = input_tensors[0];
    tt::tt_metal::Shape input_shape = input_tensor.get_legacy_shape();
    TT_FATAL(this->repeat_dim < input_shape.rank(), "Repeat dim specified is larger than input tensor rank.");
    if (input_tensor.get_layout() == Layout::ROW_MAJOR && this->repeat_dim == input_shape.rank() - 1) {
        TT_FATAL(
            (input_shape[this->repeat_dim] * input_tensor.element_size()) % ADDRESS_ALIGNMENT == 0,
            "Current repeat implementation requires aligned last dim when repeating on last dim");
    }
    TT_FATAL(this->num_repeats > 0, "Number of repeats should be greater than 0");
    TT_FATAL(input_tensor.buffer(), "Operand to repeat needs to be allocated in a buffer on device.");
    TT_FATAL(input_tensor.device(), "Operand to repeat needs to be on device.");
    TT_FATAL(
        input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Input to repeat must be interleaved.");
    TT_FATAL(
        this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Output of repeat must be interleaved.");
}

std::vector<tt::tt_metal::Shape> Repeat::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    tt::tt_metal::Shape shape_out = input_tensors[0].get_legacy_shape();
    shape_out[this->repeat_dim] *= this->num_repeats;
    return {shape_out};
}

std::vector<Tensor> Repeat::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const Tensor &ref_in_tensor = input_tensors[0];

    return operation::generic_create_output_tensors(
        *this, input_tensors, ref_in_tensor.get_dtype(), ref_in_tensor.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks Repeat::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    switch (this->get_parallelization_strategy(input_tensors)) {
        case RepeatOpParallelizationStrategy::MULTI_CORE:
            return repeat_multi_core(input_tensors[0], this->repeat_dim, this->num_repeats, output_tensors[0]);
        case RepeatOpParallelizationStrategy::SINGLE_CORE:
        default: return repeat_single_core(input_tensors[0], this->repeat_dim, this->num_repeats, output_tensors[0]);
    };
}

Tensor repeat(const Tensor &input_tensor, const Shape &shape, const MemoryConfig &output_mem_config) {
    uint32_t input_rank = input_tensor.get_legacy_shape().rank();
    TT_FATAL(shape.rank() == input_rank, "Number of repeat dims must be equal to number of tensor dims");
    Tensor output = input_tensor;
    for (uint32_t dim = 0; dim < shape.rank(); ++dim) {
        if (shape[dim] == 1) {
            continue;
        }
        TT_FATAL(shape[dim] > 0, "Number of repetitions along a dim must be greater than 0");
        if (input_tensor.get_layout() == Layout::ROW_MAJOR && dim == input_rank - 1) {
            TT_FATAL(
                (input_tensor.get_legacy_shape()[dim] * input_tensor.element_size()) % ADDRESS_ALIGNMENT == 0,
                "Current repeat implementation requires aligned last dim when repeating on last dim");
        }
        output = operation::run_without_autoformat(Repeat{dim, shape[dim], output_mem_config}, {output}).at(0);
    }
    return output;
}

}  // namespace tt_metal

}  // namespace tt
