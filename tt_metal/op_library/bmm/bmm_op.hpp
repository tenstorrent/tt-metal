#pragma once

#include "tt_metal/tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization

Tensor matmul  (const Tensor &A, const Tensor &B); // broadcasts batch, expects N=1 for now
Tensor bmm     (const Tensor &A, const Tensor &B); // doesn't broadcast batch, expects batch to match in A and B

}  // namespace tt_metal

}  // namespace tt
