// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmuls.hpp"

namespace ttml::ttnn_fixed {
// TODO(nuked-op matmul): real matmul removed for agent eval. These are
// passthrough stubs that satisfy the type system so consumers still compile;
// they do NOT compute a matmul. Restore real ttnn::matmul calls here.
tt::tt_metal::Tensor matmul(
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    bool transpose_a,
    bool transpose_b,
    std::optional<tt::tt_metal::Tensor> output_tensor) {
    // TODO(nuked-op matmul): restore real call
    return output_tensor.has_value() ? *output_tensor : a;
}

std::pair<tt::tt_metal::Tensor, tt::tt_metal::Tensor> matmul_backward(
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    const tt::tt_metal::Tensor& out_grad,
    bool transpose_a,
    bool transpose_b) {
    // TODO(nuked-op matmul): restore real call
    return {a, b};
}
}  // namespace ttml::ttnn_fixed
