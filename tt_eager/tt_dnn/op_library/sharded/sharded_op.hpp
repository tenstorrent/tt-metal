// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

enum class ShardedOpParallelizationStrategy { MULTI_CORE = 0 };

enum class ShardedOpType { InterleavedToSharded, ShardedToInterleaved };

operation::ProgramWithCallbacks interleaved_to_sharded_multi_core(
    const Tensor &a, Tensor &output, const CoreCoord &grid_size);
operation::ProgramWithCallbacks sharded_to_interleaved_multi_core(
    const Tensor &a, Tensor &output, const CoreCoord &grid_size);

struct Sharded {
    const CoreCoord grid_size;
    const ShardSpec shard_spec;
    const ShardedOpType sharded_op_type;
    const MemoryConfig output_mem_config;
    const DataType output_dtype;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    ShardedOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
    std::string get_type_name() const;

    static constexpr auto attribute_names =
        std::make_tuple("grid_size", "shard_spec", "sharded_op_type", "output_mem_config", "output_dtype");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->grid_size),
            std::cref(this->shard_spec),
            std::cref(this->sharded_op_type),
            std::cref(this->output_mem_config),
            std::cref(this->output_dtype));
    }
};

inline Tensor interleaved_to_sharded(
    const Tensor &input_tensor,
    CoreCoord grid_size,
    std::array<uint32_t, 2> shard_shape,
    TensorMemoryLayout shard_scheme,
    ShardOrientation shard_orientation,
    std::optional<const DataType> output_dtype = std::nullopt) {
    uint32_t num_cores = 0;
    uint32_t total_height = input_tensor.volume() / input_tensor.shape()[-1];
    uint32_t total_width = input_tensor.shape()[-1];
    switch (shard_scheme) {
        case TensorMemoryLayout::HEIGHT_SHARDED: num_cores = div_up(total_height, shard_shape[0]); break;
        case TensorMemoryLayout::WIDTH_SHARDED: num_cores = div_up(total_width, shard_shape[1]); break;
        case TensorMemoryLayout::BLOCK_SHARDED:
            num_cores = div_up(total_height, shard_shape[0]) * div_up(total_width, shard_shape[1]);
            break;
        default: TT_ASSERT(false, "Unsupported sharding scheme");
    }
    bool row_wise = shard_orientation == ShardOrientation::ROW_MAJOR;
    CoreRangeSet grid = num_cores_to_corerange_set(num_cores, grid_size, row_wise);
    auto shard_spec = ShardSpec{.shard_grid = grid, .shard_shape = shard_shape, .shard_orientation = shard_orientation};
    MemoryConfig sharded_mem_config = MemoryConfig{.memory_layout = shard_scheme, .buffer_type = BufferType::L1};
    return operation::run(
               Sharded{
                   .grid_size = grid_size,
                   .shard_spec = shard_spec,
                   .sharded_op_type = ShardedOpType::InterleavedToSharded,
                   .output_mem_config = sharded_mem_config,
                   .output_dtype = output_dtype.value_or(input_tensor.dtype())},
               {input_tensor})
        .at(0);
}


inline Tensor interleaved_to_sharded_core_range_set(
    const Tensor &input_tensor,
    CoreRangeSet grid,
    std::array<uint32_t, 2> shard_shape,
    TensorMemoryLayout shard_scheme,
    ShardOrientation shard_orientation,
    std::optional<const DataType> output_dtype = std::nullopt)
{

//TODO : extend for CoreRangeSets with multiple core ranges
    auto core_range = *grid.ranges().begin();
    auto grid_size = CoreCoord(core_range.end.x - core_range.start.x, core_range.end.y- core_range.start.y);
    return interleaved_to_sharded(input_tensor, grid_size, shard_shape, shard_scheme, shard_orientation, output_dtype);
}


inline Tensor sharded_to_interleaved(
    const Tensor &input_tensor,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt) {
    TT_FATAL(input_tensor.shard_spec().has_value());
    auto shard_spec = input_tensor.shard_spec().value();
    return operation::run(
               Sharded{
                   .grid_size = input_tensor.device()->compute_with_storage_grid_size(),
                   .shard_spec = shard_spec,
                   .sharded_op_type = ShardedOpType::ShardedToInterleaved,
                   .output_mem_config = output_mem_config,
                   .output_dtype = output_dtype.value_or(input_tensor.dtype())},
               {input_tensor})
        .at(0);
}

}  // namespace tt_metal

}  // namespace tt
