#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization
Tensor permute(const Tensor &a, uint32_t N, uint32_t C, uint32_t H, uint32_t W);

}  // namespace tt_metal

}  // namespace tt
