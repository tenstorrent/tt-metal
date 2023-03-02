#pragma once

#include "tt_metal/tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization
Tensor transpose(const Tensor &a);
Tensor transpose_wh_multi_core(const Tensor &a);
Tensor transpose_hc(const Tensor &a);

}  // namespace tt_metal

}  // namespace tt
