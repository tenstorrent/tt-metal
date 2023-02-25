#pragma once

#include "ll_buda/tensor/tensor.hpp"

namespace tt {

namespace ll_buda {

// TODO: Accept parallelization

Tensor matmul  (const Tensor &A, const Tensor &B); // broadcasts batch, expects N=1 for now
Tensor bmm     (const Tensor &A, const Tensor &B); // doesn't broadcast batch, expects batch to match in A and B

}  // namespace ll_buda

}  // namespace tt
