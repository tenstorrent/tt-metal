// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gram_polynomial.hpp"

#include "device/gram_polynomial_device_operation.hpp"
#include "metal/ops/gram_matmul/gram_matmul.hpp"
#include "ttnn/operations/trace.hpp"
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

namespace {

// Helper: run one iteration writing to pre-allocated buffers
void run_iteration(
    const ttnn::Tensor& X_in,
    const ttnn::Tensor& G,
    const ttnn::Tensor& H,
    const ttnn::Tensor& X_out,
    float a,
    float b,
    float c,
    const std::optional<const ttml::metal::ops::gram_polynomial::device::GramPolynomialConfig>& config,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    std::optional<const tt::tt_metal::DataType> dtype,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    ttml::metal::gram_matmul(X_in, std::nullopt, std::nullopt, std::nullopt, std::nullopt, G);
    ttnn::prim::ttml_gram_polynomial_phase2(G, b, c, config, memory_config, dtype, compute_kernel_config, H);
    ttnn::prim::ttml_hx_plus_ax(H, X_in, a, config, memory_config, dtype, compute_kernel_config, X_out);
}

}  // namespace

ttnn::Tensor newton_schulz(
    const ttnn::Tensor& x_tensor,
    float a,
    float b,
    float c,
    int num_iterations,
    bool use_trace,
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

    if (!use_trace) {
        // Non-traced path: ping-pong with pre-allocated buffers
        int src = 1;
        for (int i = 1; i < num_iterations; i++) {
            int dst = 1 - src;
            run_iteration(X[src], G, H, X[dst], a, b, c, config, memory_config, dtype, compute_kernel_config);
            src = dst;
        }

        X[1 - src].deallocate();
        G.deallocate();
        H.deallocate();
        return X[src];
    }

    // Traced path: capture two traces for ping-pong, then replay
    auto* device = x_tensor.device();
    const std::optional<ttnn::QueueId> cq_id = std::nullopt;

    // Capture trace_a: X[1] -> X[0]
    auto trace_a = ttnn::operations::trace::begin_trace_capture(device, cq_id);
    run_iteration(X[1], G, H, X[0], a, b, c, config, memory_config, dtype, compute_kernel_config);
    ttnn::operations::trace::end_trace_capture(device, trace_a, cq_id);
    // trace_a capture executed iteration 1, result in X[0]

    // Capture trace_b: X[0] -> X[1]
    auto trace_b = ttnn::operations::trace::begin_trace_capture(device, cq_id);
    run_iteration(X[0], G, H, X[1], a, b, c, config, memory_config, dtype, compute_kernel_config);
    ttnn::operations::trace::end_trace_capture(device, trace_b, cq_id);
    // trace_b capture executed iteration 2, result in X[1]

    // Iterations 1 and 2 already executed during capture. Replay for iterations 3..N-1.
    // After capture: iteration 2 result is in X[1]. Next needs trace_a (X[1]->X[0]).
    int src = 1;
    for (int i = 3; i < num_iterations; i++) {
        if (src == 1) {
            ttnn::operations::trace::execute_trace(device, trace_a, cq_id, false);
            src = 0;
        } else {
            ttnn::operations::trace::execute_trace(device, trace_b, cq_id, false);
            src = 1;
        }
    }

    ttnn::operations::trace::release_trace(device, trace_a);
    ttnn::operations::trace::release_trace(device, trace_b);

    X[1 - src].deallocate();
    G.deallocate();
    H.deallocate();
    return X[src];
}

}  // namespace ttml::metal
