// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/paged_update_cache/paged_update_cache_op.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {
namespace operations {
namespace primary {


void PagedUpdateCache::validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    const auto& cache_tensor = input_tensors.at(0);
    const auto& input_tensor = input_tensors.at(1);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE and cache_tensor.storage_type() == StorageType::DEVICE, "Operands to update_cache need to be on device!");
    TT_FATAL(input_tensor.device() == cache_tensor.device(), "Operands to update_cache need to be on the same device!");
    TT_FATAL(input_tensor.buffer() != nullptr and cache_tensor.buffer() != nullptr, "Operands to update_cache need to be allocated in buffers on device!");
    TT_FATAL((input_tensor.get_layout() == Layout::TILE && cache_tensor.get_layout() == Layout::TILE), "Inputs to update_cache must be tilized");
    TT_FATAL(input_tensor.get_dtype() == DataType::FLOAT32 || input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::BFLOAT8_B);
    TT_FATAL(cache_tensor.get_dtype() == DataType::FLOAT32 || cache_tensor.get_dtype() == DataType::BFLOAT16 || cache_tensor.get_dtype() == DataType::BFLOAT8_B);

    // input_tensor: [1, b, padded_heads, head_dim]
    // cache_tensor: [b, 1, kv_len, head_dim]
    TT_FATAL(input_tensor.get_legacy_shape()[-1] == cache_tensor.get_legacy_shape()[-1]);
    TT_FATAL(input_tensor.get_legacy_shape()[0] == 1);
    TT_FATAL(cache_tensor.get_legacy_shape()[1] == 1, "Only supports 1 head now.");
    TT_FATAL(input_tensor.get_legacy_shape()[1] == cache_tensor.get_legacy_shape()[0]);
    TT_FATAL(cache_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);

    TT_FATAL(this->op_type == PagedUpdateCacheOpType::UPDATE, "Only UPDATE operation is supported for PagedUpdateCache");

    if (this->op_type == PagedUpdateCacheOpType::UPDATE) {
        TT_FATAL(optional_input_tensors.at(0).has_value() != this->update_idxs.size() > 0, "Only an update tensor or an update vector can be provided. Not both or neither.");

        uint32_t num_indices;
        if (optional_input_tensors.at(0).has_value()) {
            const auto& update_idxs_tensor = optional_input_tensors.at(0).value();
            TT_FATAL(update_idxs_tensor.get_dtype() == DataType::INT32);
            TT_FATAL(update_idxs_tensor.get_layout() == Layout::ROW_MAJOR);
            // update_idxs_tensor: [num_indices]
            num_indices = update_idxs_tensor.get_legacy_shape()[0];

            // must be iterleaved
            TT_FATAL(update_idxs_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);
            // must be in dram
            TT_FATAL(update_idxs_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM);
        } else {
            num_indices = this->update_idxs.size();
        }
        TT_FATAL(input_tensor.get_legacy_shape()[1] == num_indices, "Number of update_idxs should match batch size");
        TT_FATAL(input_tensor.is_sharded());
        if (input_tensor.is_sharded()) {
            TT_FATAL(input_tensor.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED);
            TT_FATAL(input_tensor.shard_spec().value().shape[1] == input_tensor.get_legacy_shape()[-1]);
            // Require even work division for now
            TT_FATAL(input_tensor.shard_spec().value().grid.num_cores() == cache_tensor.get_legacy_shape()[0], "Input must be sharded on batch num cores");
            TT_FATAL((input_tensor.volume() / input_tensor.get_legacy_shape()[-1]) % input_tensor.shard_spec().value().shape[0] == 0);
            TT_FATAL(input_tensor.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR, "Only ROW_MAJOR sharding is supported");
        }
        TT_FATAL(cache_tensor.get_legacy_shape()[0] <= input_tensor.get_legacy_shape()[-1]);
        // batch offset is only valid if num_user less than 32 and batch_offset + num_user <= 32
        if (cache_tensor.get_legacy_shape()[0] < 32) TT_FATAL(this->batch_offset + cache_tensor.get_legacy_shape()[0] <= 32);
        else TT_FATAL(this->batch_offset == 0);
    }
}

std::vector<Shape> PagedUpdateCache::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    // Do nothing because it's an in-place operation
    return {};
}

std::vector<Tensor> PagedUpdateCache::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    // Do nothing because it's an in-place operation
    return {};
}

operation::ProgramWithCallbacks PagedUpdateCache::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& cache_tensor = input_tensors.at(0);
    const auto& input_tensor = input_tensors.at(1);
    const auto update_idxs_tensor = optional_input_tensors.at(0); // TODO: Is this tensor passed around by value?


    switch(this->get_parallelization_strategy(input_tensors)) {
        case PagedUpdateCacheOpParallelizationStrategy::MULTI_CORE:
        default:
            return paged_update_cache_multi_core(cache_tensor, input_tensor, update_idxs_tensor, this->update_idxs, this->batch_offset, this->compute_kernel_config);
    };
}


PagedUpdateCacheOpParallelizationStrategy PagedUpdateCache::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    return PagedUpdateCacheOpParallelizationStrategy::MULTI_CORE;
}

const operation::Hash PagedUpdateCache::compute_program_hash(
    const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    return operation::hash_operation<PagedUpdateCache>(this->op_type, input_tensors);
}

}   // namespace primary
}   // namespace operations
}   // namespace tt
