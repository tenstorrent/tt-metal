// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "log_softmax.hpp"

#include "device/log_softmax_device_operation.hpp"
#include "device/row_softmax_program.hpp"
#include "metal/common/tensor_validation.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"

namespace ttml::metal {

bool log_softmax_supported(const ttnn::Tensor& input) {
    return input.logical_shape().rank() == 4 && input.logical_volume() != 0 && input.layout() == ttnn::Layout::TILE &&
           input.memory_config() == ttnn::DRAM_MEMORY_CONFIG &&
           ops::row_softmax::supports_tile(input.tensor_spec().tile()) && has_only_tile_padding(input) &&
           (input.dtype() == ttnn::DataType::BFLOAT16 || input.dtype() == ttnn::DataType::FLOAT32);
}

std::vector<ttnn::Tensor> log_softmax_fw(const ttnn::Tensor& input) {
    return ttnn::prim::ttml_log_softmax_fw(input);
}

ttnn::Tensor log_softmax_bw(const ttnn::Tensor& input, const ttnn::Tensor& upstream, const ttnn::Tensor& state) {
    // The kernels read the gradient tile for tile alongside the input. A gradient may arrive elsewhere (an L1 tensor
    // passed to set_grad, say), so bring it to the input's layout and placement; FP32 and BF16 are read as they are.
    ttnn::Tensor grad = upstream;
    if (grad.dtype() != ttnn::DataType::BFLOAT16 && grad.dtype() != ttnn::DataType::FLOAT32) {
        grad = ttnn::typecast(grad, input.dtype());
    }
    if (grad.layout() != ttnn::Layout::TILE) {
        grad = ttnn::to_layout(grad, ttnn::Layout::TILE);
    }
    if (grad.memory_config() != input.memory_config()) {
        grad = ttnn::to_memory_config(grad, input.memory_config());
    }
    return ttnn::prim::ttml_log_softmax_bw(input, grad, state);
}

}  // namespace ttml::metal
