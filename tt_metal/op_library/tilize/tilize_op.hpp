#pragma once

#include "tt_metal/tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization

Tensor tilize (const Tensor &a);
Tensor tilize_with_zero_padding (const Tensor &a);
Tensor tilize_conv_activation (const Tensor &a);
}  // namespace tt_metal

}  // namespace tt
