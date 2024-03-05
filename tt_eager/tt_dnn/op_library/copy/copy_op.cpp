// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/copy/copy_op.hpp"

#include "tt_metal/host_api.hpp"

#include "third_party/magic_enum/magic_enum.hpp"


namespace tt {

namespace tt_metal {

void Copy::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to copy need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to copy need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Copy does not currently support sharding");
    if (input_tensors.size() == 2) {
        const auto& dst_tensor = input_tensors[1];
        TT_FATAL(input_tensor_a.get_legacy_shape() == dst_tensor.get_legacy_shape());
        TT_FATAL(input_tensor_a.get_layout() == dst_tensor.get_layout());
        TT_FATAL(input_tensor_a.memory_config().memory_layout == dst_tensor.memory_config().memory_layout);
        TT_FATAL(dst_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Copy does not currently support sharding");
    }
    if (this->output_dtype != input_tensor_a.get_dtype()) {
        TT_FATAL(input_tensor_a.get_layout() == Layout::TILE, "Only tile layout supports dtype conversion");
    }
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Copy does not currently support sharding");
}

std::vector<Shape> Copy::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    if (input_tensors.size() == 2) {
        return {input_tensors[1].get_legacy_shape()};
    } else {
        const auto& input_tensor = input_tensors.at(0);
        return {input_tensor.get_legacy_shape()};
    }
}

std::vector<Tensor> Copy::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    if (input_tensors.size() == 2) {
        return {input_tensors[1]};
    } else {
        const auto& input_tensor = input_tensors.at(0);
        return operation::generic_create_output_tensors(*this, input_tensors, this->output_dtype, input_tensor.get_layout(), this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Copy::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);

    switch (Copy::get_parallelization_strategy(input_tensors)){
        case CopyOpParallelizationStrategy::MULTI_CORE:
            return copy_multi_core(input_tensor, output_tensor);
        case CopyOpParallelizationStrategy::SINGLE_CORE:
        default:
            return copy_single_core(input_tensor, output_tensor);
    }
}

CopyOpParallelizationStrategy Copy::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    uint32_t num_units = input_tensor.get_layout() == Layout::TILE ? input_tensor.volume() / TILE_HW : input_tensor.volume() / input_tensor.get_legacy_shape()[-1];
    if (num_units > 1) {
        return CopyOpParallelizationStrategy::MULTI_CORE;
    }
    else{
        return CopyOpParallelizationStrategy::SINGLE_CORE;
    }
}

tt::stl::reflection::Attributes Copy::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
        {"output_dtype", this->output_dtype}
    };
}

Tensor copy(const Tensor& src_tensor, const Tensor& dst_tensor) {
    operation::run(Copy{dst_tensor.memory_config(), dst_tensor.get_dtype()}, {src_tensor, dst_tensor});
    return dst_tensor;
}

Tensor clone(const Tensor& input, const MemoryConfig& output_mem_config, std::optional<const DataType> output_dtype) {
    return operation::run(Copy{output_mem_config, output_dtype.value_or(input.get_dtype())}, {input}).at(0);
}

Tensor typecast(const Tensor& input_tensor, const DataType& dtype, const MemoryConfig& output_mem_config ) {
    return operation::run(Copy{output_mem_config, dtype}, {input_tensor}).at(0);
}

//unary assign
Tensor assign(const Tensor& input, const MemoryConfig& output_mem_config , std::optional<const DataType> output_dtype ) {
    return operation::run(Copy{output_mem_config, output_dtype.value_or(input.get_dtype())}, {input}).at(0);
}

// binary assign
Tensor assign(const Tensor& input_a, const Tensor& input_b) {
    operation::run(Copy{input_b.memory_config(), input_b.get_dtype()}, {input_a, input_b});
    return input_b;
}

}  // namespace tt_metal

}  // namespace tt
