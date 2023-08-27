#pragma once

#include "tt_dnn/op_library/run_operation.hpp"
#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization
Tensor permute(const Tensor &a, uint32_t N, uint32_t C, uint32_t H, uint32_t W, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt_metal

}  // namespace tt
