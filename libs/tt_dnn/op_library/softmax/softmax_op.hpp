#pragma once

#include "libs/tensor/tensor.hpp"

namespace tt { namespace tt_metal {

Tensor softmax(const Tensor &a);

} }  // namespace tt::tt_metal
