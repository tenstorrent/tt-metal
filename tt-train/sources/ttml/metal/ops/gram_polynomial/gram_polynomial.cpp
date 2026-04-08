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
    // Kernel computes G² + (b/c)*G (unscaled). Host applies c scaling after.
    // This avoids SFPU overhead in the kernel that causes either:
    //   - 80% slowdown (scaling in K-loop stalls FPU pipeline), or
    //   - subs>1 deadlock (scaling in reduce phase blocks K-delivery)
    operation_attributes_t attrs{.b = b, .c = c, .output_mode = output_mode, .math_fidelity = math_fidelity};
    auto result = ttnn::device_operation::launch<GramPolynomialDeviceOperation>(
        attrs, tensor_args_t{.input_tensor = G, .preallocated_output = preallocated_output});
    if (c != 1.0f) {
        result = ttnn::multiply(result, c);
    }
    return result;
}

}  // namespace ttml::metal
