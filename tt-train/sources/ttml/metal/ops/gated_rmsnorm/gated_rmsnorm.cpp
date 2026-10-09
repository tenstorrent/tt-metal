// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_rmsnorm.hpp"

#include "core/compute_kernel_config.hpp"
#include "device/gated_rmsnorm_bw_device_operation.hpp"
#include "device/gated_rmsnorm_fw_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor gated_rmsnorm_fw(
    const ttnn::Tensor& input, const ttnn::Tensor& gate, const ttnn::Tensor& gamma, const float epsilon) {
    return ttnn::prim::ttml_gated_rmsnorm_fw(input, gate, gamma, epsilon);
}

std::tuple<ttnn::Tensor, ttnn::Tensor, std::optional<ttnn::Tensor>> gated_rmsnorm_bw(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& gamma,
    const ttnn::Tensor& dL_dout,
    const float epsilon,
    const bool compute_dgamma) {
    auto outputs = ttnn::prim::ttml_gated_rmsnorm_bw(input, gate, gamma, dL_dout, epsilon, compute_dgamma);

    std::optional<ttnn::Tensor> dgamma = std::nullopt;
    if (compute_dgamma) {
        // The kernel emits x * inv * du per element; gamma is shared by all rows and heads, so sum
        // [B, 1, T, H*V] -> [1, 1, 1, H*V] -> [1, 1, H, V] -> [1, 1, 1, V].
        const uint32_t width = input.logical_shape()[-1];
        const uint32_t group = gamma.logical_shape()[-1];
        auto per_column = ttnn::sum(
            outputs[2].value(),
            ttsl::SmallVector<int>{0, 1, 2},
            /* keep_dim */ true,
            /* output_mem_config */ std::nullopt,
            core::ComputeKernelConfig::precise());
        auto per_head = ttnn::reshape(per_column, ttnn::Shape({1U, 1U, width / group, group}));
        dgamma = ttnn::sum(
            per_head,
            ttsl::SmallVector<int>{2},
            /* keep_dim */ true,
            /* output_mem_config */ std::nullopt,
            core::ComputeKernelConfig::precise());
    }
    return {outputs[0].value(), outputs[1].value(), dgamma};
}

}  // namespace ttml::metal
