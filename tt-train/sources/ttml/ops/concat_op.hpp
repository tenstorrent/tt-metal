#pragma once

#include <vector>

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr concat(const std::vector<autograd::TensorPtr>& tensors, int32_t dim);

}
