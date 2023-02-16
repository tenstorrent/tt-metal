#pragma once

#include "ll_buda/tensor/tensor.hpp"

namespace tt {

namespace ll_buda {

// TODO: Accept parallelization
// TODO(AP): merge with transpose
Tensor transpose(const Tensor &a);

}  // namespace ll_buda

}  // namespace tt
