// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw.hpp"

#include "core/compute_kernel_config.hpp"
#include "device/rmsnorm_bw_device_operation.hpp"

namespace ttml::metal {

namespace {

using ops::rmsnorm_bw::device::RMSNormBackwardMode;

// dL_dgamma requires a sum over batches and rows, which the kernel leaves to one ttnn::sum: either over the
// per-row components [B,1,S,C] or over the per-core fp32 partials [1,1,num_cores*32,C].
ttnn::Tensor reduce_dgamma(const ttnn::Tensor& dgamma_parts, const ttnn::Tensor& gamma) {
    auto dgamma = ttnn::sum(
        dgamma_parts,
        /* dim_arg */ ttsl::SmallVector<int>{0, 1, 2},
        /* keep_dim */ true,
        /* output_mem_config */ std::nullopt,
        /*compute_kernel_config */ core::ComputeKernelConfig::precise());  // -> [1,1,1,C]
    if (dgamma.dtype() != gamma.dtype()) {
        dgamma = ttnn::typecast(dgamma, gamma.dtype());
    }
    return dgamma;
}

}  // namespace

std::vector<std::optional<ttnn::Tensor>> rmsnorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,
    const ttnn::Tensor& dL_dout_tensor) {
    auto result = ttnn::prim::ttml_rmsnorm_bw(
        input_tensor,   // [B,1,S,C]
        gamma_tensor,   // [1,1,1,C]
        rms_tensor,     //[B,1,S,1]
        dL_dout_tensor  //[B,1,S,C]
    );
    return {result[0], reduce_dgamma(result[1], gamma_tensor)};
}

ttnn::Tensor rmsnorm_bw_local_sum(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& gamma_tensor, const ttnn::Tensor& dL_dout_tensor) {
    auto result = ttnn::prim::ttml_rmsnorm_bw(
        input_tensor,
        gamma_tensor,
        /* rms */ std::nullopt,
        dL_dout_tensor,
        /* epsilon */ 1e-6F,
        /* preallocated_da */ std::nullopt,
        /* preallocated_dgamma_components */ std::nullopt,
        RMSNormBackwardMode::LocalSum);
    return result[0];
}

std::vector<std::optional<ttnn::Tensor>> rmsnorm_bw_external_scale(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const ttnn::Tensor& gathered_local_sums,
    uint32_t full_width) {
    auto result = ttnn::prim::ttml_rmsnorm_bw(
        input_tensor,
        gamma_tensor,
        rms_tensor,
        dL_dout_tensor,
        /* epsilon */ 1e-6F,
        /* preallocated_da */ std::nullopt,
        /* preallocated_dgamma_components */ std::nullopt,
        RMSNormBackwardMode::ExternalScale,
        full_width,
        gathered_local_sums);
    return {result[0], reduce_dgamma(result[1], gamma_tensor)};
}

}  // namespace ttml::metal
