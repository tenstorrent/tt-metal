#pragma once

#include "tensor/tensor.hpp"

namespace tt {

namespace tt_metal {

// TODO: Accept parallelization

Tensor untilize (const Tensor &a);
}  // namespace tt_metal

}  // namespace tt
