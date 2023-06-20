#pragma once

#include "tensor/tensor.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

enum class BertLargeTMOpType {
    SPLIT_FUSED_QKV = 0,
    CREATE_Q_HEAD = 1,
    CREATE_K_HEAD = 2,
    CREATE_V_HEAD = 3,
    CONCAT_HEADS = 4,
};

Program multi_core_split_fused_qkv(const Tensor &input_tensor_a, std::vector<Tensor> &output, CoreCoord compute_and_storage_grid_size);
Program multi_core_create_qkv_heads(const Tensor &input_tensor_a, Tensor &output_tensor, CoreCoord compute_and_storage_grid_size, bool transpose_hw);
Program multi_core_concat_heads(const Tensor &input_tensor_a, Tensor &output_tensor, CoreCoord compute_and_storage_grid_size);

struct BertLargeTM {
    BertLargeTMOpType bert_large_tm_op_type;
    MemoryConfig output_mem_config;

    void validate(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const;
};

inline std::vector<Tensor> bert_large_split_fused_qkv(const Tensor &input_tensor_a, const MemoryConfig& mem_config) {
    return operation::run(BertLargeTM{BertLargeTMOpType::SPLIT_FUSED_QKV, mem_config}, {std::cref(input_tensor_a)});
}
inline Tensor bert_large_create_q_head(const Tensor &input_tensor_a, const MemoryConfig& mem_config) {
    return std::move(operation::run(BertLargeTM{BertLargeTMOpType::CREATE_Q_HEAD, mem_config}, {std::cref(input_tensor_a)}).at(0));
}
inline Tensor bert_large_create_k_head(const Tensor &input_tensor_a, const MemoryConfig& mem_config) {
    return std::move(operation::run(BertLargeTM{BertLargeTMOpType::CREATE_K_HEAD, mem_config}, {std::cref(input_tensor_a)}).at(0));
}
inline Tensor bert_large_create_v_head(const Tensor &input_tensor_a, const MemoryConfig& mem_config) {
    return std::move(operation::run(BertLargeTM{BertLargeTMOpType::CREATE_V_HEAD, mem_config}, {std::cref(input_tensor_a)}).at(0));
}
inline Tensor bert_large_concat_heads(const Tensor &input_tensor_a, const MemoryConfig& mem_config) {
    return std::move(operation::run(BertLargeTM{BertLargeTMOpType::CONCAT_HEADS, mem_config}, {std::cref(input_tensor_a)}).at(0));
}

}  // namespace tt_metal

}  // namespace tt
