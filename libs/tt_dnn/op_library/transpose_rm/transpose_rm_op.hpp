#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// TODO(AP): need to merge transpose with other formats
Tensor transpose_hc_rm (const Tensor &a);
Tensor transpose_hc_rm_multi_core(const Tensor &a);

}  // namespace tt_metal

}  // namespace tt
