#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

std::vector<Tensor> multi_core_split_fused_qkv(const Tensor &a, const MemoryConfig& mem_config, CoreCoord compute_and_storage_grid_size);
Tensor multi_core_create_qkv_heads(const Tensor &a, const MemoryConfig& mem_config, CoreCoord compute_and_storage_grid_size, bool transpose_hw);

std::vector<Tensor> bert_large_split_fused_qkv(const Tensor &a, const MemoryConfig& mem_config);
Tensor bert_large_create_q_head(const Tensor &a, const MemoryConfig& mem_config);
Tensor bert_large_create_k_head(const Tensor &a, const MemoryConfig& mem_config);
Tensor bert_large_create_v_head(const Tensor &a, const MemoryConfig& mem_config);

}  // namespace tt_metal

}  // namespace tt
