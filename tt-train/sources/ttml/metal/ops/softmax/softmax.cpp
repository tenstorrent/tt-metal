// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax.hpp"

namespace ttml::metal {

ttnn::Tensor softmax(const ttnn::Tensor& input_tensor, int32_t dim) {
    throw std::runtime_error("softmax operation has been removed");
}

}  // namespace ttml::metal
