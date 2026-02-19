// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gram_polynomial.hpp"

#include "device/gram_polynomial_device_operation.hpp"
#include "metal/ops/gram_matmul/gram_matmul.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

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

ttnn::Tensor newton_schulz_iteration(
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

ttnn::Tensor newton_schulz(
    const ttnn::Tensor& x_tensor,
    float a,
    float b,
    float c,
    int num_iterations,
    const std::optional<const ttml::metal::ops::gram_polynomial::device::GramPolynomialConfig>& config,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<const tt::tt_metal::DataType> dtype,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    if (num_iterations <= 0) {
        return x_tensor;
    }

    // First iteration: allocates G, H, and X[1]. These become our reusable buffers.
    auto G = ttml::metal::gram_matmul(x_tensor);
    auto H = ttnn::prim::ttml_gram_polynomial_phase2(G, b, c, config, memory_config, dtype, compute_kernel_config);

    ttnn::Tensor X[2];
    X[0] = x_tensor;
    X[1] = ttnn::prim::ttml_hx_plus_ax(H, X[0], a, config, memory_config, dtype, compute_kernel_config);

    if (num_iterations == 1) {
        G.deallocate();
        H.deallocate();
        return X[1];
    }

    // Allocate X[0] buffer for ping-pong (same spec as X[1], no computation)
    X[0] = create_device_tensor(X[1].tensor_spec(), x_tensor.device());

    // Remaining iterations: ping-pong X[src] -> X[dst], reuse G and H buffers
    int src = 1;
    for (int i = 1; i < num_iterations; i++) {
        int dst = 1 - src;
        ttml::metal::gram_matmul(X[src], std::nullopt, std::nullopt, std::nullopt, std::nullopt, G);
        ttnn::prim::ttml_gram_polynomial_phase2(G, b, c, config, memory_config, dtype, compute_kernel_config, H);
        ttnn::prim::ttml_hx_plus_ax(H, X[src], a, config, memory_config, dtype, compute_kernel_config, X[dst]);
        src = dst;
    }

    X[1 - src].deallocate();
    G.deallocate();
    H.deallocate();
    return X[src];
}

}  // namespace ttml::metal
