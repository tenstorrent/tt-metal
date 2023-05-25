#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

std::vector<Tensor> multi_core_split_fused_qkv(const Tensor &a, const MemoryConfig& mem_config, CoreCoord compute_and_storage_grid_size);

std::vector<Tensor> bert_large_split_fused_qkv(const Tensor &a, const MemoryConfig& mem_config);

}  // namespace tt_metal

}  // namespace tt
