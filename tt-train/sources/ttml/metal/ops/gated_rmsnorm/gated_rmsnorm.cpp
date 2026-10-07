// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gated_rmsnorm.hpp"

#include "core/compute_kernel_config.hpp"
#include "device/gated_rmsnorm_bw_device_operation.hpp"
#include "device/gated_rmsnorm_fw_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor gated_rmsnorm_fw(
    const ttnn::Tensor& input, const ttnn::Tensor& gate, const ttnn::Tensor& gamma, float epsilon) {
    return ttnn::prim::ttml_gated_rmsnorm_fw(input, gate, gamma, epsilon);
}

std::vector<std::optional<ttnn::Tensor>> gated_rmsnorm_bw(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& gamma,
    const ttnn::Tensor& dL_dout,
    float epsilon,
    bool compute_dgamma) {
    auto result = ttnn::prim::ttml_gated_rmsnorm_bw(input, gate, gamma, dL_dout, epsilon, compute_dgamma);
    std::vector<std::optional<ttnn::Tensor>> out{result[0], result[1], std::nullopt};
    if (compute_dgamma) {
        // dgamma_components is [B,1,T,W]; reduce over tokens, then fold the W = num_groups*group axis onto
        // the group axis so every head shares one gamma.
        const auto& comp = result[2].value();
        const uint32_t width = comp.logical_shape()[-1];
        const uint32_t group = gamma.logical_shape()[-1];
        auto per_width = ttnn::sum(
            comp,
            /* dim_arg */ ttsl::SmallVector<int>{0, 1, 2},
            /* keep_dim */ true,
            /* output_mem_config */ std::nullopt,
            /* compute_kernel_config */ core::ComputeKernelConfig::precise());  // [1,1,1,W]
        auto grouped = ttnn::reshape(per_width, ttnn::Shape{1, 1, width / group, group});
        out[2] = ttnn::sum(
            grouped,
            /* dim_arg */ ttsl::SmallVector<int>{2},
            /* keep_dim */ true,
            /* output_mem_config */ std::nullopt,
            /* compute_kernel_config */ core::ComputeKernelConfig::precise());  // [1,1,1,group]
    }
    return out;
}

}  // namespace ttml::metal
