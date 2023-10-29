// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/sharded/sharded_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

void Sharded::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr , "Operands to shard need to be allocated in buffers on device!");
    // TT_ASSERT(input_tensor.dtype() == DataType::BFLOAT16);
    if (this->sharded_op_type == ShardedOpType::InterleavedToSharded) {
        TT_ASSERT(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
    } else if (this->sharded_op_type == ShardedOpType::ShardedToInterleaved) {
        TT_ASSERT(input_tensor.memory_config().is_sharded());
        if(input_tensor.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
            if (input_tensor.shape()[-1] % this->shard_spec.shard_shape[1] != 0
                || ((input_tensor.volume() / input_tensor.shape()[-1]) % this->shard_spec.shard_shape[0]) != 0) {
                TT_ASSERT(input_tensor.shard_spec().value().shard_grid.ranges().size() == 1);
            }
        }
    }
    auto device_grid = input_tensor.device()->compute_with_storage_grid_size();
    TT_ASSERT(this->grid_size.x <= device_grid.x && this->grid_size.y <= device_grid.y);
    // Divisibility of num_cores and shard size with tensor shape is done in tensor creation, so no need to assert here
}

std::vector<Shape> Sharded::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.shape()};
}

std::vector<Tensor> Sharded::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->sharded_op_type == ShardedOpType::InterleavedToSharded) {
        return {create_sharded_device_tensor(this->compute_output_shapes(input_tensors).at(0), input_tensor.dtype(), input_tensor.layout(), input_tensor.device(), this->output_mem_config, this->shard_spec)};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.dtype(), input_tensor.layout(), this->output_mem_config);
    }
}

operation::ProgramWithCallbacks Sharded::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    if (this->sharded_op_type == ShardedOpType::InterleavedToSharded) {
        return interleaved_to_sharded_multi_core(input_tensor, output_tensor, this->grid_size);
    } else {
        return sharded_to_interleaved_multi_core(input_tensor, output_tensor, this->grid_size);
    }
}

std::string Sharded::get_type_name() const {
    return magic_enum::enum_name(this->sharded_op_type).data();
}

ShardedOpParallelizationStrategy Sharded::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    return ShardedOpParallelizationStrategy::MULTI_CORE;
}

tt::stl::reflection::Attributes Sharded::attributes() const {
    return {
        {"shard_spec", this->shard_spec},
        {"sharded_op_type", this->sharded_op_type},
        {"output_mem_config", this->output_mem_config},
    };
}


}  // namespace tt_metal

}  // namespace tt
