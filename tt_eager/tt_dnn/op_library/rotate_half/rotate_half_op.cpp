// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/rotate_half/rotate_half_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {


void RotateHalf::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to rotate half need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to rotate half need to be allocated in buffers on device!");
    TT_FATAL((input_tensor.get_layout() == Layout::TILE), "Inputs to rotate half must be tilized");
    TT_FATAL(input_tensor.get_legacy_shape()[-1] % (TILE_WIDTH * 2) == 0, "Input X dim must be divisible into tiles");
    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "RotateHalf does not currently support sharding");
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "RotateHalf does not currently support sharding");
}

std::vector<Shape> RotateHalf::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}

std::vector<Tensor> RotateHalf::create_output_tensors(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks RotateHalf::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    return rotate_half_single_core(input_tensor, output_tensor);
}


RotateHalfOpParallelizationStrategy RotateHalf::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    uint32_t num_rows = input_tensor.volume() / input_tensor.get_legacy_shape()[-1] / TILE_HEIGHT;
    return RotateHalfOpParallelizationStrategy::SINGLE_CORE;
}

tt::stl::reflection::Attributes RotateHalf::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
    };
}

}  // namespace tt_metal

}  // namespace tt
