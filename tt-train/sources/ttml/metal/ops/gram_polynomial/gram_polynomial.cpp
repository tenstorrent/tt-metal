// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gram_polynomial.hpp"

#include <ttnn/device_operation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "device/gram_polynomial_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor gram_polynomial(
    const ttnn::Tensor& G,
    float b,
    float c,
    OutputMode output_mode,
    MathFidelity math_fidelity,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using namespace ops::gram_polynomial::device;
    operation_attributes_t attrs{.b = b, .c = c, .output_mode = output_mode, .math_fidelity = math_fidelity};
    return ttnn::device_operation::launch<GramPolynomialDeviceOperation>(
        attrs, tensor_args_t{.input_tensor = G, .preallocated_output = preallocated_output});
}

}  // namespace ttml::metal
