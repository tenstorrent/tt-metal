// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/clone/clone_op.hpp"

#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"


namespace tt {

namespace tt_metal {

void Clone::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to clone need to be on device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands to clone need to be allocated in buffers on device!");
    TT_ASSERT((input_tensor_a.layout() == Layout::TILE), "Inputs to clone must be tilized");
}

std::vector<Shape> Clone::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> Clone::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks Clone::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    switch (Clone::get_parallelization_strategy(input_tensors)){
        case CloneOpParallelizationStrategy::MULTI_CORE:
            return clone_multi_core(input_tensor, output_tensor);
        case CloneOpParallelizationStrategy::SINGLE_CORE:
        default:
            return clone_single_core(input_tensor, output_tensor);
    }
}

CloneOpParallelizationStrategy Clone::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    uint32_t num_tiles = input_tensor.volume() / TILE_HW;
    if (num_tiles > 1) {
        return CloneOpParallelizationStrategy::MULTI_CORE;
    }
    else{
        return CloneOpParallelizationStrategy::SINGLE_CORE;
    }
}

tt::stl::reflection::Attributes Clone::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config}
    };
}

}  // namespace tt_metal

}  // namespace tt
