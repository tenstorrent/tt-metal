#pragma once

#include <core/ttnn_all_includes.hpp>

#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr view(const autograd::TensorPtr& tensor, const ttnn::Shape& shape);

}
