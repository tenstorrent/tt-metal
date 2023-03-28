#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization

Tensor tilize (const Tensor &a);
Tensor tilize_with_zero_padding (const Tensor &a);
Tensor tilize_conv_activation (const Tensor &a, bool conv1x1 = false);
}  // namespace tt_metal

}  // namespace tt
