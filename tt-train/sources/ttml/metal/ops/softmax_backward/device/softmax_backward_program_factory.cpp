// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_backward_program_factory.hpp"

namespace ttml::metal::ops::softmax_backward::device {

SoftmaxBackwardFactory::cached_program_t SoftmaxBackwardFactory::create(
    const SoftmaxBackwardParams &, const SoftmaxBackwardInputs &, ttnn::Tensor &) {
    TT_THROW("softmax_backward program factory has been removed");
}

void SoftmaxBackwardFactory::override_runtime_arguments(
    cached_program_t &, const SoftmaxBackwardParams &, const SoftmaxBackwardInputs &, ttnn::Tensor &) {
    TT_THROW("softmax_backward program factory has been removed");
}

}  // namespace ttml::metal::ops::softmax_backward::device
