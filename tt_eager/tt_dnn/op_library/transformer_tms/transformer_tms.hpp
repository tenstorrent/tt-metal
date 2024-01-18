// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

namespace transformers {

operation::ProgramWithCallbacks multi_core_split_query_key_value_and_split_heads(const Tensor &input_tensor, std::vector<Tensor> &output, CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_split_query_key_value_and_split_heads_sharded(const Tensor &input_tensor, std::vector<Tensor> &output, CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_concat_heads(const Tensor &input_tensor, Tensor &output_tensor, CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_attn_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor &output_tensor, std::optional<const uint32_t> num_tokens, std::optional<const bool> transpose_hw, CoreCoord compute_with_storage_grid_size);

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

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

inline Tensor attn_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const CoreCoord& compute_with_storage_grid_size, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype=std::nullopt) {
    return operation::run(AttnMatmul{std::nullopt, std::nullopt, compute_with_storage_grid_size, mem_config, output_dtype.value_or(input_tensor_a.dtype())}, {input_tensor_a, input_tensor_b}).at(0);
}

inline Tensor attn_matmul_from_cache(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const uint32_t num_tokens, const bool transpose_hw, const CoreCoord& compute_with_storage_grid_size, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype=std::nullopt) {
    TT_ASSERT(num_tokens > 0, "Number of tokens must be at least 1!");
    TT_ASSERT(num_tokens <= input_tensor_b.shape()[2], "Number of tokens must be smaller or equal to the max cache length (B.shape[2])!");
    const uint32_t num_tokens_rounded_up_to_32 = ((num_tokens - 1) / 32 + 1) * 32;
    return operation::run(AttnMatmul{num_tokens_rounded_up_to_32, transpose_hw, compute_with_storage_grid_size, mem_config, output_dtype.value_or(input_tensor_a.dtype())}, {input_tensor_a, input_tensor_b}).at(0);
}

}  // namespace transformers

}  // namespace primary

}  // namespace operations

}  // namespace tt
