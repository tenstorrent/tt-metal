#pragma once

#include "ll_buda/tensor/tensor.hpp"

namespace tt {

namespace ll_buda {

// Tensor &a cannot be const, since in some cases we modify in place
Tensor reshape (Tensor &a, int N, int C, int H, int W);

}  // namespace ll_buda

}  // namespace tt
