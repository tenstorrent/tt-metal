#pragma once

#include "tensor/tensor.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

namespace transformers {

operation::ProgramWithCallbacks multi_core_split_fused_qkv_and_split_heads(const Tensor &input_tensor, std::vector<Tensor> &output, CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_concat_heads(const Tensor &input_tensor, Tensor &output_tensor, CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_attn_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, Tensor &output_tensor, CoreCoord compute_with_storage_grid_size, DataType output_dtype);

struct SplitFusedQKVAndSplitHeads {
    CoreCoord compute_with_storage_grid_size;
    MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

inline std::tuple<Tensor, Tensor, Tensor> split_fused_qkv_and_split_heads(const Tensor &input_tensor, const CoreCoord& compute_with_storage_grid_size, const MemoryConfig& mem_config) {
    auto output_tensors = operation::run(SplitFusedQKVAndSplitHeads{compute_with_storage_grid_size, mem_config}, {input_tensor});
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
    CoreCoord compute_with_storage_grid_size;
    MemoryConfig output_mem_config;
    DataType output_dtype;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

inline Tensor attn_matmul(const Tensor &input_tensor_a, const Tensor &input_tensor_b, const CoreCoord& compute_with_storage_grid_size, const MemoryConfig& mem_config, std::optional<const DataType> output_dtype=std::nullopt) {
    return operation::run(AttnMatmul{compute_with_storage_grid_size, mem_config, output_dtype.value_or(input_tensor_a.dtype())}, {input_tensor_a, input_tensor_b}).at(0);
}

}  // namespace transformers

}  // namespace primary

}  // namespace operations

}  // namespace tt
