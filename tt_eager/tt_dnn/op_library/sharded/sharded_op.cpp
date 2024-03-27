// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/sharded/sharded_op.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

void Sharded::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
    if (this->sharded_op_type == ShardedOpType::InterleavedToSharded) {
        TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
        if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
            TT_FATAL(this->shard_spec.shape[1] * input_tensor.element_size() % L1_ALIGNMENT == 0, "Shard page size must currently have L1 aligned page size");
        }
    } else if (this->sharded_op_type == ShardedOpType::ShardedToInterleaved) {
        TT_FATAL(input_tensor.memory_config().is_sharded());
        if (input_tensor.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
            if (input_tensor.get_legacy_shape()[-1] % this->shard_spec.shape[1] != 0 ||
                ((input_tensor.volume() / input_tensor.get_legacy_shape()[-1]) % this->shard_spec.shape[0]) != 0) {
                TT_FATAL(input_tensor.shard_spec().value().grid.ranges().size() == 1);
            }
        }
        if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
            TT_FATAL(this->shard_spec.shape[1] * input_tensor.element_size() % (this->output_mem_config.buffer_type == BufferType::DRAM ? DRAM_ALIGNMENT : L1_ALIGNMENT) == 0, "Shard page size must be aligned to output buffer type alignment");
        }
    }
    if (input_tensor.get_dtype() != this->output_dtype) {
        TT_FATAL(input_tensor.get_layout() == Layout::TILE);
    }
    auto device_grid = input_tensor.device()->compute_with_storage_grid_size();
    TT_FATAL(this->grid_size.x <= device_grid.x && this->grid_size.y <= device_grid.y);
    // Divisibility of num_cores and shard size with tensor shape is done in tensor creation, so no need to assert here
}

std::vector<Shape> Sharded::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}

std::vector<Tensor> Sharded::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->sharded_op_type == ShardedOpType::InterleavedToSharded) {
        auto mem_config = this->output_mem_config;
        mem_config.shard_spec = this->shard_spec;
        return {create_sharded_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            this->output_dtype,
            input_tensor.get_layout(),
            input_tensor.device(),
            mem_config,
            true
            )};
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, this->output_dtype, input_tensor.get_layout(), this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Sharded::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    if (this->sharded_op_type == ShardedOpType::InterleavedToSharded) {
        return interleaved_to_sharded_multi_core(input_tensor, output_tensor);
    } else {
        return sharded_to_interleaved_multi_core(input_tensor, output_tensor);
    }
}

std::string Sharded::get_type_name() const { return magic_enum::enum_name(this->sharded_op_type).data(); }

ShardedOpParallelizationStrategy Sharded::get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const {
    return ShardedOpParallelizationStrategy::MULTI_CORE;
}


void Reshard::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.is_sharded(), "input must be sharded");
    TT_FATAL(this->output_mem_config.is_sharded(), "output must be sharded");
    if(input_tensor.get_layout() == Layout::ROW_MAJOR) {
        bool same_row_size = input_tensor.memory_config().shard_spec.value().shape[1] == this->output_mem_config.shard_spec.value().shape[1];
        TT_FATAL(same_row_size, "row major must have shard_spec[1] be the same on both input and output");
    }
}

std::vector<Shape> Reshard::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}

operation::ProgramWithCallbacks Reshard::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {

    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    //each tensor has its respective shard_spec within its memory_config
    return reshard_multi_core(input_tensor, output_tensor);


}

std::vector<Tensor> Reshard::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto mem_config = this->output_mem_config;




    return {create_sharded_device_tensor(
        this->compute_output_shapes(input_tensors).at(0),
        input_tensor.get_dtype(),
        input_tensor.get_layout(),
        input_tensor.device(),
        mem_config,
        true
        )};
}


ShardedOpParallelizationStrategy Reshard::get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const {
    return ShardedOpParallelizationStrategy::MULTI_CORE;
}

}  // namespace tt_metal

}  // namespace tt
