/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>

#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

enum class ShardedOpType {
    INTERLEAVED_TO_SHARDED, SHARDED_TO_INTERLEAVED
};

operation::ProgramWithCallbacks interleaved_to_sharded_multi_core(const Tensor &a, Tensor &output);
operation::ProgramWithCallbacks sharded_to_interleaved_multi_core(const Tensor &a, Tensor &output);

struct Sharded {
    const ShardSpec shard_spec;
    const ShardedOpType sharded_op_type;
    const MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

inline Tensor interleaved_to_sharded(const Tensor &input_tensor, uint32_t num_cores, std::pair<uint32_t, uint32_t> shard_shape, TensorMemoryLayout shard_scheme) {
    CoreRangeSet grid = num_cores_to_corerange_set(num_cores, input_tensor.device()->compute_with_storage_grid_size(), true);
    auto shard_spec = ShardSpec{.shard_grid=grid, .shard_shape=shard_shape};
    MemoryConfig sharded_mem_config = MemoryConfig{.memory_layout = shard_scheme, .buffer_type = BufferType::L1};
    return operation::run(Sharded{.shard_spec=shard_spec, .sharded_op_type=ShardedOpType::INTERLEAVED_TO_SHARDED, sharded_mem_config}, {input_tensor}).at(0);
}

inline Tensor sharded_to_interleaved(const Tensor &input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_ASSERT(input_tensor.memory_config().is_sharded());
    auto shard_spec = input_tensor.shard_spec().value();
    return operation::run(Sharded{.shard_spec=shard_spec, .sharded_op_type=ShardedOpType::SHARDED_TO_INTERLEAVED, .output_mem_config=output_mem_config}, {input_tensor}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
