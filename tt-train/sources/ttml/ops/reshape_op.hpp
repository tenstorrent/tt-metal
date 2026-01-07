#pragma once

#include <core/ttnn_all_includes.hpp>

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr reshape_op(const autograd::TensorPtr& tensor, const ttnn::Shape& new_shape);

}
