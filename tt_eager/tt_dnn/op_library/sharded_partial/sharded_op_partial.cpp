// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/sharded_partial/sharded_op_partial.hpp"

#include "tt_metal/common/assert.hpp"
#include "tensor/types.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

void ShardedPartial::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    // Validate output tensor
    TT_FATAL(slice_index >= 0 && slice_index < num_slices, "Slice index and num_slices don't match! Index = {} num_slices = {}", slice_index, num_slices);
    TT_FATAL(input_tensor.get_layout() == Layout::TILE, "Currently, only tile layout is supported for partial I->S");
    TT_FATAL((input_tensor.volume() / input_tensor.get_legacy_shape()[-1]) % num_slices == 0, "Total height of a tensor must be divisible by num_slices!");

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
    if (this->sharded_op_type == ShardedOpPartialType::InterleavedToShardedPartial) {
        TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
    } else if (this->sharded_op_type == ShardedOpPartialType::ShardedToInterleavedPartial) {
        TT_FATAL(input_tensor.memory_config().is_sharded());
        if (input_tensor.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
            if (input_tensor.get_legacy_shape()[-1] % this->shard_spec.shape[1] != 0 ||
                ((input_tensor.volume() / input_tensor.get_legacy_shape()[-1]) % this->shard_spec.shape[0]) != 0) {
                TT_FATAL(input_tensor.shard_spec().value().grid.ranges().size() == 1);
            }
        }
    }
    if (input_tensor.get_dtype() != this->output_dtype) {
        TT_FATAL(input_tensor.get_layout() == Layout::TILE);
    }
    auto device_grid = input_tensor.device()->compute_with_storage_grid_size();
    TT_FATAL(this->grid_size.x <= device_grid.x && this->grid_size.y <= device_grid.y);
    // Divisibility of num_cores and shard size with tensor shape is done in tensor creation, so no need to assert here
}

std::vector<Shape> ShardedPartial::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    Shape shape = input_tensor.get_legacy_shape();

    // For I->S, output shapes will be different because of slicing (and need to be divided)
    if (this->sharded_op_type == ShardedOpPartialType::InterleavedToShardedPartial) {
        uint32_t total_height = input_tensor.volume() / shape[-1];
        uint32_t new_height = total_height / this->num_slices;

        shape[0] = 1;
        shape[1] = 1;
        shape[2] = new_height;
    }

    // If S->I, in input_tensors[1] we will have cache tensor
    if (this->sharded_op_type == ShardedOpPartialType::ShardedToInterleavedPartial) {
        const auto& output_tensor = input_tensors.at(1);
        return {output_tensor.get_legacy_shape()};
    }

    return {shape};
}

std::vector<Tensor> ShardedPartial::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->sharded_op_type == ShardedOpPartialType::InterleavedToShardedPartial) {
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
        // Don't create anything, we already passed in output tensor
        return {};
    }
}

operation::ProgramWithCallbacks ShardedPartial::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    if (this->sharded_op_type == ShardedOpPartialType::InterleavedToShardedPartial) {
        auto& output_tensor = output_tensors.at(0);
        return interleaved_to_sharded_partial_multi_core(input_tensor, output_tensor, this->num_slices, this->slice_index);
    } else {
        // For S->I partial, we store output tensor as input_tensors[1]
        auto& output_tensor = input_tensors[1];
        return sharded_to_interleaved_partial_multi_core(input_tensor, output_tensor, this->num_slices, this->slice_index);
    }
}

std::string ShardedPartial::get_type_name() const { return magic_enum::enum_name(this->sharded_op_type).data(); }

ShardedOpPartialParallelizationStrategy ShardedPartial::get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const {
    return ShardedOpPartialParallelizationStrategy::MULTI_CORE;
}

}  // namespace tt_metal

}  // namespace tt
