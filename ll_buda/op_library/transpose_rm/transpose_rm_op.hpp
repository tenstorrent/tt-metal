#pragma once

#include "ll_buda/tensor/tensor.hpp"

namespace tt {

namespace ll_buda {

// TODO(AP): need to merge transpose with other formats
Tensor transpose_hc_rm (const Tensor &a);

}  // namespace ll_buda

}  // namespace tt
