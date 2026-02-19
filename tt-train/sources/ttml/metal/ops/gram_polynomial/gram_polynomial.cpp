// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gram_polynomial.hpp"

#include "device/gram_polynomial_device_operation.hpp"
#include "metal/ops/gram_matmul/gram_matmul.hpp"

namespace ttml::metal {

ttnn::Tensor gram_polynomial(
    const ttnn::Tensor& input_tensor,
    float b,
    float c,
    const std::optional<const ttml::metal::ops::gram_polynomial::device::GramPolynomialConfig>& config,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<const tt::tt_metal::DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    // Phase 1: G = X @ X^T (existing gram_matmul op)
    auto G = ttml::metal::gram_matmul(input_tensor);

    // Phase 2: H = c*G*G + b*G (new device op)
    auto H = ttnn::prim::ttml_gram_polynomial_phase2(G, b, c, config, memory_config, dtype, compute_kernel_config);

    G.deallocate();
    return H;
}

ttnn::Tensor muon_precondition(
    const ttnn::Tensor& x_tensor,
    float a,
    float b,
    float c,
    const std::optional<const ttml::metal::ops::gram_polynomial::device::GramPolynomialConfig>& config,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<const tt::tt_metal::DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    // Phase 1: G = X @ X^T
    auto G = ttml::metal::gram_matmul(x_tensor);

    // Phase 2: H = c*G*G + b*G
    auto H = ttnn::prim::ttml_gram_polynomial_phase2(G, b, c, config, memory_config, dtype, compute_kernel_config);
    G.deallocate();

    // Phase 3: X' = H @ X + a*X
    auto X_prime = ttnn::prim::ttml_hx_plus_ax(H, x_tensor, a, config, memory_config, dtype, compute_kernel_config);
    H.deallocate();

    return X_prime;
}

}  // namespace ttml::metal
