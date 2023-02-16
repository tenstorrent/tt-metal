#pragma once

#include "ll_buda/tensor/tensor.hpp"

namespace tt {

namespace ll_buda {

// TODO: Accept parallelization

Tensor matmul (const Tensor &A, const Tensor &B);

}  // namespace ll_buda

}  // namespace tt
