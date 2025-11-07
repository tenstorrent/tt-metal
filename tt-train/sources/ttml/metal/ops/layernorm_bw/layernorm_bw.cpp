// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_bw.hpp"

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "device/layernorm_bw_device_operation.hpp"

namespace ttml::metal::ops::layernorm_bw {

std::vector<std::optional<ttnn::Tensor>> LayerNormBackwardOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& x_hat_tensor,
    const ttnn::Tensor& rstd_tensor,
    const ttnn::Tensor& dL_dout_tensor) {
    // Call the device operation
    // Returns: [dx, dgamma_components, dbeta_components]
    auto result = ttnn::prim::ttml_layernorm_bw(input_tensor, gamma_tensor, x_hat_tensor, rstd_tensor, dL_dout_tensor);

    // dL_dgamma and dL_dbeta require sum over batches so we cannot perform this sum in the kernel.
    // Instead we return the components and reduce them here.
    ttml::autograd::ctx().get_profiler().read_results(
        &autograd::ctx().get_device(), "dgamma and dbeta reductions started");

    return {
        result[0],  // dx - already complete
        ttnn::reshape(
            ttnn::sum(
                ttnn::reshape(
                    result[1],
                    ttnn::Shape(
                        {result[1].logical_shape()[0] * result[1].logical_shape()[1] * result[1].logical_shape()[2],
                         result[1].logical_shape()[3]})),  // dgamma_components
                /* dim_arg */ ttnn::SmallVector<int>{0},
                /* keep_dim */ true,
                /* output_mem_config */ std::nullopt,
                /*compute_kernel_config */ core::ComputeKernelConfig::precise()),
            ttnn::Shape({1, 1, 1, result[1].logical_shape()[3]})),
        ttnn::reshape(
            ttnn::sum(
                ttnn::reshape(
                    result[2],
                    ttnn::Shape(
                        {result[2].logical_shape()[0] * result[2].logical_shape()[1] * result[2].logical_shape()[2],
                         result[2].logical_shape()[3]})),  // dbeta_components
                /* dim_arg */ ttnn::SmallVector<int>{0},
                /* keep_dim */ true,
                /* output_mem_config */ std::nullopt,
                /*compute_kernel_config */ core::ComputeKernelConfig::precise()),
            ttnn::Shape({1, 1, 1, result[2].logical_shape()[3]}))};
}

}  // namespace ttml::metal::ops::layernorm_bw
