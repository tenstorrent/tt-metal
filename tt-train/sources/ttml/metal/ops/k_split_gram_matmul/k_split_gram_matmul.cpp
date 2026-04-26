// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "k_split_gram_matmul.hpp"

#include <ttnn/device_operation.hpp>

#include "device/k_split_gram_matmul_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor gram_matmul(
    const ttnn::Tensor& input,
    OutputMode output_mode,
    MathFidelity math_fidelity,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using namespace ops::k_split_gram_matmul::device;
    operation_attributes_t attrs{.output_mode = output_mode, .math_fidelity = math_fidelity};
    return ttnn::device_operation::launch<KSplitGramMatmulDeviceOperation>(
        attrs, tensor_args_t{.input_tensor = input, .preallocated_output = preallocated_output});
}

}  // namespace ttml::metal
