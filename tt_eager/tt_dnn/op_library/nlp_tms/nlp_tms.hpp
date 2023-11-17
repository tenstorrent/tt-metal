// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor/tensor.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

enum class NlpTMOpType {
    CREATE_QKV_HEADS = 0,
    CONCAT_HEADS = 1,
};

operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads(const Tensor &input_tensor_a, std::vector<Tensor> &output, CoreCoord compute_with_storage_grid_size);
operation::ProgramWithCallbacks multi_core_nlp_concat_heads(const Tensor &input_tensor_a, Tensor &output, CoreCoord compute_with_storage_grid_size);

struct NlpTM {
    NlpTMOpType nlp_tm_op_type;
    MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
};

inline std::vector<Tensor> nlp_create_qkv_heads(const Tensor &input_tensor_a, const MemoryConfig& mem_config) {
    // TODO: Uplift to support generic qkv num_heads and head_dim; currently, hard-coded for falcon-7b
    return operation::run(NlpTM{NlpTMOpType::CREATE_QKV_HEADS, mem_config}, {input_tensor_a});
}
inline Tensor nlp_concat_heads(const Tensor &input_tensor_a, const MemoryConfig& mem_config) {
    // TODO: Uplift to support generic num_heads and head_dim; currently, hard-coded for falcon-7b
    return operation::run(NlpTM{NlpTMOpType::CONCAT_HEADS, mem_config}, {input_tensor_a}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
