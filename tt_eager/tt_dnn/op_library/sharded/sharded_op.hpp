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

operation::ProgramWithCallbacks interleaved_to_sharded_multi_core(const Tensor &a, const Tensor &output, uint32_t num_slices = 1, uint32_t slice_index = 0);
operation::ProgramWithCallbacks sharded_to_interleaved_multi_core(const Tensor &a, const Tensor &output, uint32_t num_slices = 1, uint32_t slice_index = 0);
operation::ProgramWithCallbacks reshard_runtime_args_multi_core(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks reshard_config_tensor_multi_core(const Tensor& input, const Tensor &config_vector, Tensor& output);

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
    const std::variant<CoreCoord, CoreRangeSet> grid,
    const std::array<uint32_t, 2> shard_shape,
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
                uint32_t total_width = input_tensor.get_legacy_shape()[-1];
                switch (shard_scheme) {
                    case TensorMemoryLayout::HEIGHT_SHARDED: num_cores = div_up(total_height, shard_shape[0]); break;
                    case TensorMemoryLayout::WIDTH_SHARDED: num_cores = div_up(total_width, shard_shape[1]); break;
                    case TensorMemoryLayout::BLOCK_SHARDED:
                        num_cores = div_up(total_height, shard_shape[0]) * div_up(total_width, shard_shape[1]);
                        break;
                    default: TT_ASSERT(false, "Unsupported sharding scheme");
                }
                grid_set = num_cores_to_corerange_set(num_cores, grid_size, row_wise);
            } else if constexpr (std::is_same_v<GridType, CoreRangeSet>) {
                auto bbox = grid.bounding_box();
                grid_size = CoreCoord{bbox.end.x + 1, bbox.end.y + 1};
                grid_set = grid;
            }
        },
        grid);
    ShardSpec shard_spec(grid_set, shard_shape, shard_orientation);
    MemoryConfig sharded_mem_config = MemoryConfig{.memory_layout = shard_scheme, .buffer_type = BufferType::L1};
    return operation::run(
               Sharded{
                   .grid_size = grid_size,
                   .shard_spec = shard_spec,
                   .sharded_op_type = ShardedOpType::InterleavedToSharded,
                   .output_mem_config = sharded_mem_config,
                   .output_dtype = output_dtype.value_or(input_tensor.get_dtype())},
               {input_tensor})
        .at(0);
}

// start is inclusive, end is exclusive
struct PageRange {
    uint32_t start;
    uint32_t end;
};

struct CorePageRange {
    CoreCoord core;
    PageRange range;
};

std::unordered_map<CoreCoord, std::vector<CorePageRange>> get_core_page_ranges(
    Buffer* input_buffer, Buffer* output_buffer);

enum class ReshardRunTimeArgType { RUNTIME_ARGS, CONFIG_TENSOR };
// TODO: tarafdarTT unify with Sharded struct
struct Reshard {
    const MemoryConfig output_mem_config;
    const ReshardRunTimeArgType rt_type;
    std::unordered_map<CoreCoord, std::vector<CorePageRange>> output_core_to_page_range_pair;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    ShardedOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("output_mem_config");
    const auto attribute_values() const { return std::make_tuple(std::cref(this->output_mem_config)); }
};

inline Tensor interleaved_to_sharded(
    const Tensor &input_tensor,
    const MemoryConfig &sharded_mem_config,
    std::optional<const DataType> output_dtype = std::nullopt) {
    TT_FATAL(sharded_mem_config.is_sharded());
    auto bbox = sharded_mem_config.shard_spec.value().grid.bounding_box();
    CoreCoord grid_size(bbox.end.x + 1, bbox.end.y + 1);
    return operation::run(
               Sharded{
                   .grid_size = grid_size,
                   .shard_spec = sharded_mem_config.shard_spec.value(),
                   .sharded_op_type = ShardedOpType::InterleavedToSharded,
                   .output_mem_config = sharded_mem_config,
                   .output_dtype = output_dtype.value_or(input_tensor.get_dtype())},
               {input_tensor})
        .at(0);
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
                   .output_dtype = output_dtype.value_or(input_tensor.get_dtype())},
               {input_tensor})
        .at(0);
}

inline Tensor reshard(const Tensor &input_tensor, const MemoryConfig &output_mem_config) {
    TT_FATAL(input_tensor.shard_spec().has_value());
    TT_FATAL(output_mem_config.is_sharded());

    return operation::run(Reshard{.output_mem_config = output_mem_config, .rt_type= ReshardRunTimeArgType::RUNTIME_ARGS}, {input_tensor}).at(0);
}


inline Tensor reshard_config_tensor(const Tensor &input_tensor, const MemoryConfig &output_mem_config) {
    TT_FATAL(input_tensor.shard_spec().has_value());
    TT_FATAL(output_mem_config.is_sharded());

    return operation::run(Reshard{.output_mem_config = output_mem_config,  .rt_type= ReshardRunTimeArgType::CONFIG_TENSOR}, {input_tensor}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
