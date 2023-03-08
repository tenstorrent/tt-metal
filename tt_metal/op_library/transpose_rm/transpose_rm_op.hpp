#pragma once

#include "tt_metal/tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// TODO(AP): need to merge transpose with other formats
Tensor transpose_hc_rm (const Tensor &a);

}  // namespace tt_metal

}  // namespace tt
