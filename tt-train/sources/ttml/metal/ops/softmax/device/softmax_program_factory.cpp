// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_program_factory.hpp"

namespace ttml::metal::ops::softmax::device {

SoftmaxProgramFactory::cached_program_t SoftmaxProgramFactory::create(
    const operation_attributes_t &, const tensor_args_t &, tensor_return_value_t &) {
    TT_THROW("softmax program factory has been removed");
}

void SoftmaxProgramFactory::override_runtime_arguments(
    cached_program_t &, const operation_attributes_t &, const tensor_args_t &, tensor_return_value_t &) {
    TT_THROW("softmax program factory has been removed");
}

}  // namespace ttml::metal::ops::softmax::device
