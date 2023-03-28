#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

Tensor pad_h_rm (const Tensor &a, int paddedH);

}  // namespace tt_metal

}  // namespace tt
