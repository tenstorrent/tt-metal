// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/typecast/typecast_op.hpp"

#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"


namespace tt {

namespace tt_metal {

void Typecast::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to copy need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to copy need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
    if (input_tensors.size() == 2) {
        const auto& dst_tensor = input_tensors[1];
        TT_FATAL(input_tensor_a.shape() == dst_tensor.shape());
        TT_FATAL(input_tensor_a.layout() == dst_tensor.layout());
        TT_FATAL(input_tensor_a.dtype() == dst_tensor.dtype());
        TT_FATAL(input_tensor_a.memory_config().memory_layout == dst_tensor.memory_config().memory_layout);
    }
}

std::vector<Shape> Typecast::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    if (input_tensors.size() == 2) {
        return {};
    } else {
        const auto& input_tensor = input_tensors.at(0);
        return {input_tensor.shape()};
    }
}

std::vector<Tensor> Typecast::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    if (input_tensors.size() == 2) {
        return {};
    } else {
        const auto& input_tensor = input_tensors.at(0);
        return operation::generic_create_output_tensors(*this, input_tensors, tt::tt_metal::DataType::BFLOAT8_B, input_tensor.layout(), this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Typecast::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = input_tensors.size() == 2 ? input_tensors[1] : output_tensors.at(0);

    switch (Typecast::get_parallelization_strategy(input_tensors)){
        case TypecastOpParallelizationStrategy::MULTI_CORE:
            return typecast_multi_core(input_tensor, output_tensor);
        case TypecastOpParallelizationStrategy::SINGLE_CORE:
        default:
            return typecast_single_core(input_tensor, output_tensor);
    }
}

TypecastOpParallelizationStrategy Typecast::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    uint32_t num_units = input_tensor.layout() == Layout::TILE ? input_tensor.volume() / TILE_HW : input_tensor.volume() / input_tensor.shape()[-1];
    if (num_units > 1) {
        return TypecastOpParallelizationStrategy::MULTI_CORE;
    }
    else{
        return TypecastOpParallelizationStrategy::SINGLE_CORE;
    }
}

tt::stl::reflection::Attributes Typecast::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config}
    };
}

}  // namespace tt_metal

}  // namespace tt
