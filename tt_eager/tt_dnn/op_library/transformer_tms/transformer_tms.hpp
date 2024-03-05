// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

#include "tt_dnn/op_library/compute_kernel_config.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

namespace transformers {

operation::ProgramWithCallbacks multi_core_split_query_key_value_and_split_heads(const Tensor &input_tensor, std::vector<Tensor> &output, CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_split_query_key_value_and_split_heads_sharded(const Tensor &input_tensor, std::vector<Tensor> &output, CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_concat_heads(const Tensor &input_tensor, Tensor &output_tensor, CoreCoord compute_with_storage_grid_size);
// TODO: Group attention matmul will support sharding, mcasting, and should be faster; we should make attn_matmul (ie. KV heads = 1) a special case of group_attn_matmul and run the same op
operation::ProgramWithCallbacks multi_core_attn_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor &output_tensor, std::optional<const uint32_t> num_tokens, std::optional<const bool> transpose_hw, CoreCoord compute_with_storage_grid_size, DeviceComputeKernelConfig compute_kernel_config);
operation::ProgramWithCallbacks multi_core_group_attn_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor &output_tensor, std::optional<const uint32_t> num_tokens, std::optional<const bool> transpose_hw, const uint32_t out_subblock_w, CoreCoord compute_with_storage_grid_size, const bool row_major, DeviceComputeKernelConfig compute_kernel_config);

struct SplitFusedQKVAndSplitHeads {
    CoreCoord compute_with_storage_grid_size;
    MemoryConfig output_mem_config;
    uint32_t num_heads;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

inline std::tuple<Tensor, Tensor, Tensor> split_query_key_value_and_split_heads(const Tensor &input_tensor, const CoreCoord& compute_with_storage_grid_size, const MemoryConfig& mem_config, const uint32_t num_heads = 16) {
    auto output_tensors = operation::run(SplitFusedQKVAndSplitHeads{compute_with_storage_grid_size, mem_config, num_heads}, {input_tensor});
    return {output_tensors.at(0), output_tensors.at(1), output_tensors.at(2)};
}

struct ConcatenateHeads {
    CoreCoord compute_with_storage_grid_size;
    MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

inline Tensor concatenate_heads(const Tensor &input_tensor, const CoreCoord& compute_with_storage_grid_size, const MemoryConfig& mem_config) {
    return operation::run(ConcatenateHeads{compute_with_storage_grid_size, mem_config}, {input_tensor}).at(0);
}

struct AttnMatmul {
    std::optional<const uint32_t> num_tokens;
    std::optional<const bool> transpose_hw;
    CoreCoord compute_with_storage_grid_size;
    MemoryConfig output_mem_config;
    DataType output_dtype;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

inline Tensor attn_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const CoreCoord& compute_with_storage_grid_size, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype=std::nullopt, std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
    auto arch = input_tensor_a.storage_type() == StorageType::DEVICE ? input_tensor_a.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config);
    return operation::run(AttnMatmul{std::nullopt, std::nullopt, compute_with_storage_grid_size, mem_config, output_dtype.value_or(input_tensor_a.get_dtype()), kernel_config_val}, {input_tensor_a, input_tensor_b}).at(0);
}

inline Tensor attn_matmul_from_cache(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const uint32_t num_tokens, const bool transpose_hw, const CoreCoord& compute_with_storage_grid_size, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype=std::nullopt, std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
    TT_FATAL(num_tokens > 0, "Number of tokens must be at least 1!");
    TT_FATAL(num_tokens <= input_tensor_b.get_legacy_shape()[2], "Number of tokens must be smaller or equal to the max cache length (B.shape[2])!");
    const uint32_t num_tokens_rounded_up_to_32 = ((num_tokens - 1) / 32 + 1) * 32;
    auto arch = input_tensor_a.storage_type() == StorageType::DEVICE ? input_tensor_a.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config);
    return operation::run(AttnMatmul{num_tokens_rounded_up_to_32, transpose_hw, compute_with_storage_grid_size, mem_config, output_dtype.value_or(input_tensor_a.get_dtype()), kernel_config_val}, {input_tensor_a, input_tensor_b}).at(0);
}

// TODO: Should we support option to read directly from cache (with optional transpose_hw)?
struct GroupAttnMatmul {
    std::optional<const uint32_t> num_tokens;
    std::optional<const bool> transpose_hw;
    const uint32_t out_subblock_w;
    CoreCoord compute_with_storage_grid_size;
    MemoryConfig output_mem_config;
    DataType output_dtype;
    const bool row_major;  // Specifies how work is distributed across cores
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

inline Tensor group_attn_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const CoreCoord& compute_with_storage_grid_size, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype=std::nullopt, std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
    bool row_major = false;
    // GroupAttnMatmul::validate will check that any sharded memory configs have same orientation
    if (input_tensor_a.is_sharded()) {
        row_major = input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
    } else if (input_tensor_b.is_sharded()) {
        row_major = input_tensor_b.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR;
    } else if (mem_config.is_sharded()) {
        if (mem_config.shard_spec.has_value()) {
            row_major = mem_config.shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        }
    }

    auto arch = input_tensor_a.storage_type() == StorageType::DEVICE ? input_tensor_a.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config);

    // Need to cache on out_subblock_w because it must be a compile time arg for optimal use of templated pack_untilize APIs
    const uint32_t Nt = input_tensor_b.get_legacy_shape()[-1] / TILE_WIDTH;
    constexpr uint32_t HALF_DST_MAX = 8; // 8 is the max number of tiles for half DST (assuming out_subblock_h == 1)
    constexpr uint32_t HALF_DST_MAX_FP32 = 4; // max dst tiles are 4 for fp32
    uint32_t out_subblock_w;

    std::visit([&](auto&& kernel_config_val) {
        using T = std::decay_t<decltype(kernel_config_val)>;
        if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            out_subblock_w = kernel_config_val.fp32_dest_acc_en ? std::min(Nt, HALF_DST_MAX_FP32) : std::min(Nt, HALF_DST_MAX);
        } else {
            out_subblock_w = std::min(Nt, HALF_DST_MAX);
        }
    }, kernel_config_val);

    return operation::run(GroupAttnMatmul{std::nullopt, std::nullopt, out_subblock_w, compute_with_storage_grid_size, mem_config, output_dtype.value_or(input_tensor_a.get_dtype()), row_major, kernel_config_val}, {input_tensor_a, input_tensor_b}).at(0);
}

}  // namespace transformers

}  // namespace primary

}  // namespace operations

}  // namespace tt
