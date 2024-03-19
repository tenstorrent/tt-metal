// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "common/assert.hpp"
#include "impl/buffers/buffer.hpp"
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

enum class ShardedOpPartialParallelizationStrategy { MULTI_CORE = 0 };

enum class ShardedOpPartialType { InterleavedToShardedPartial, ShardedToInterleavedPartial };


operation::ProgramWithCallbacks interleaved_to_sharded_partial_multi_core(const Tensor &a, const Tensor &output, int num_slices, int slice_index);
operation::ProgramWithCallbacks sharded_to_interleaved_partial_multi_core(const Tensor &a, const Tensor &output, int num_slices, int slice_index);

struct ShardedPartial {
    const CoreCoord grid_size;
    const ShardSpec shard_spec;
    const uint32_t num_slices;
    const uint32_t slice_index;
    const ShardedOpPartialType sharded_op_type;
    const MemoryConfig output_mem_config;
    const DataType output_dtype;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    ShardedOpPartialParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
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

inline Tensor interleaved_to_sharded_partial(
    const Tensor &input_tensor,
    const std::variant<CoreCoord, CoreRangeSet> grid,
    const std::array<uint32_t, 2> shard_shape,
    const uint32_t num_slices,
    const uint32_t slice_index,
    const TensorMemoryLayout shard_scheme,
    const ShardOrientation shard_orientation,
    const std::optional<const DataType> output_dtype = std::nullopt) {
    bool row_wise = shard_orientation == ShardOrientation::ROW_MAJOR;

    CoreCoord grid_size;
    CoreRangeSet grid_set({});
    std::visit(
        [&](const auto &grid) {
            using GridType = std::decay_t<decltype(grid)>;
            if constexpr (std::is_same_v<GridType, CoreCoord>) {
                grid_size = grid;
                uint32_t num_cores = 0;
                uint32_t total_height = input_tensor.volume() / input_tensor.get_legacy_shape()[-1];
                total_height /= num_slices;

                uint32_t total_width = input_tensor.get_legacy_shape()[-1];
                switch (shard_scheme) {
                    case TensorMemoryLayout::HEIGHT_SHARDED: num_cores = div_up(total_height, shard_shape[0]); break;
                    case TensorMemoryLayout::WIDTH_SHARDED: num_cores = div_up(total_width, shard_shape[1]); break;
                    case TensorMemoryLayout::BLOCK_SHARDED:
                        num_cores = div_up(total_height, shard_shape[0]) * div_up(total_width, shard_shape[1]);
                        log_info("Selected number of cores for I->S partial op is {}", num_cores);
                        break;
                    default: TT_ASSERT(false, "Unsupported sharding scheme");
                }
                grid_set = num_cores_to_corerange_set(num_cores, grid_size, row_wise);
            } else if constexpr (std::is_same_v<GridType, CoreRangeSet>) {
                TT_FATAL("Unsupported type for grid.");
            }
        },
        grid);
    ShardSpec shard_spec(grid_set, shard_shape, shard_orientation);
    MemoryConfig sharded_mem_config = MemoryConfig{.memory_layout = shard_scheme, .buffer_type = BufferType::L1};
    return operation::run(
               ShardedPartial{
                   .grid_size = grid_size,
                   .shard_spec = shard_spec,
                   .num_slices = num_slices,
                   .slice_index = slice_index,
                   .sharded_op_type = ShardedOpPartialType::InterleavedToShardedPartial,
                   .output_mem_config = sharded_mem_config,
                   .output_dtype = output_dtype.value_or(input_tensor.get_dtype())},
               {input_tensor})
        .at(0);
}

inline Tensor sharded_to_interleaved_partial(
    const Tensor &input_tensor,
    const Tensor &output_tensor,
    const uint32_t num_slices,
    const uint32_t slice_index,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DataType> output_dtype = std::nullopt) {
    TT_FATAL(input_tensor.shard_spec().has_value());
    auto shard_spec = input_tensor.shard_spec().value();
    operation::run(
               ShardedPartial{
                   .grid_size = input_tensor.device()->compute_with_storage_grid_size(),
                   .shard_spec = shard_spec,
                   .num_slices = num_slices,
                   .slice_index = slice_index,
                   .sharded_op_type = ShardedOpPartialType::ShardedToInterleavedPartial,
                   .output_mem_config = output_mem_config,
                   .output_dtype = output_dtype.value_or(input_tensor.get_dtype())},
               {input_tensor, output_tensor});
    return output_tensor;
}

}  // namespace tt_metal

}  // namespace tt
