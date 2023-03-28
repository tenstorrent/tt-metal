#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// Tensor &a cannot be const, since in some cases we modify in place
Tensor reshape (Tensor &a, int N, int C, int H, int W);

}  // namespace tt_metal

}  // namespace tt
