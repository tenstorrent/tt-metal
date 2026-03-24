// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "k_split_gram_matmul.hpp"

#include <ttnn/device_operation.hpp>

#include "device/k_split_gram_matmul_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor gram_matmul(
    const ttnn::Tensor& input, ops::k_split_gram_matmul::device::OutputMode output_mode, MathFidelity math_fidelity) {
    using namespace ops::k_split_gram_matmul::device;
    operation_attributes_t attrs{.output_mode = output_mode, .math_fidelity = math_fidelity};
    return ttnn::device_operation::launch<KSplitGramMatmulDeviceOperation>(attrs, tensor_args_t{.input_tensor = input});
}

}  // namespace ttml::metal
