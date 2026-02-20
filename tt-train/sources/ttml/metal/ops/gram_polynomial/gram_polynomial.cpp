// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gram_polynomial.hpp"

#include "device/gram_polynomial_device_operation.hpp"
#include "metal/ops/gram_matmul/gram_matmul.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/trace.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttml::metal {

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

    auto* device = x_tensor.device();

    // Iteration 0: allocates G, H, X[0] with correct shapes
    auto G = ttml::metal::gram_matmul(x_tensor);
    auto H = ttnn::prim::ttml_gram_polynomial_phase2(G, b, c, config, memory_config, dtype, compute_kernel_config);

    if (!use_trace) {
        // Non-traced path: ping-pong between X[0] and X[1], reuse G and H
        ttnn::Tensor X[2];
        X[0] = ttnn::prim::ttml_hx_plus_ax(H, x_tensor, a, config, memory_config, dtype, compute_kernel_config);

        if (num_iterations == 1) {
            G.deallocate();
            H.deallocate();
            return X[0];
        }

        X[1] = create_device_tensor(X[0].tensor_spec(), device);

        int src = 0;
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

    // Traced path: iteration 0 outside trace compiles programs,
    // then capture one iteration + copy feedback, replay N-1 times
    auto X_out = ttnn::prim::ttml_hx_plus_ax(H, x_tensor, a, config, memory_config, dtype, compute_kernel_config);

    if (num_iterations == 1) {
        G.deallocate();
        H.deallocate();
        return X_out;
    }

    auto X_in = create_device_tensor(X_out.tensor_spec(), device);
    const std::optional<ttnn::QueueId> cq_id = std::nullopt;

    ttnn::copy(X_out, X_in);

    auto tid = ttnn::operations::trace::begin_trace_capture(device, cq_id);
    ttml::metal::gram_matmul(X_in, std::nullopt, std::nullopt, std::nullopt, std::nullopt, G);
    ttnn::prim::ttml_gram_polynomial_phase2(G, b, c, config, memory_config, dtype, compute_kernel_config, H);
    ttnn::prim::ttml_hx_plus_ax(H, X_in, a, config, memory_config, dtype, compute_kernel_config, X_out);
    // TODO: This is not optimal; Could capture two traces for both addresses instead, but will be less readable
    ttnn::copy(X_out, X_in);
    ttnn::operations::trace::end_trace_capture(device, tid, cq_id);

    for (int i = 2; i < num_iterations; i++) {
        ttnn::operations::trace::execute_trace(device, tid, cq_id, false);
    }

    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    ttnn::operations::trace::release_trace(device, tid);

    X_in.deallocate();
    G.deallocate();
    H.deallocate();
    return X_out;
}

}  // namespace ttml::metal
