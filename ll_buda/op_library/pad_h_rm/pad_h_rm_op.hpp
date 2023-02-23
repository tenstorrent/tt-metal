#pragma once

#include "ll_buda/tensor/tensor.hpp"

namespace tt {

namespace ll_buda {

Tensor pad_h_rm (const Tensor &a, int paddedH);

}  // namespace ll_buda

}  // namespace tt
