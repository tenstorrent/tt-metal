// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_bw.hpp"

#include <tt-metalium/constants.hpp>
#include <vector>

#include "core/compute_kernel_config.hpp"
#include "device/polynorm_bw_device_operation.hpp"

namespace ttml::metal {

namespace {

ttnn::Tensor extract_scalar_at_column(const ttnn::Tensor& tensor, uint32_t col) {
    const ttsl::SmallVector<uint32_t> start = {0U, 0U, 0U, col};
    const ttsl::SmallVector<uint32_t> end = {1U, 1U, 1U, col + 1U};
    const ttsl::SmallVector<uint32_t> step = {1U, 1U, 1U, 1U};
    return ttnn::slice(tensor, start, end, step);
}

}  // namespace

std::tuple<std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>, std::optional<ttnn::Tensor>> polynorm3_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const ttnn::Tensor& weight_tensor,
    float epsilon,
    bool compute_grad_input,
    bool compute_grad_weight,
    bool compute_grad_bias) {
    // dw and db share the packed partials, so the kernel computes them, and the wrapper reduces them, when either is
    // requested.
    const bool compute_packed_partials = compute_grad_weight || compute_grad_bias;
    auto fused_outputs = ttnn::prim::ttml_polynorm3_bw(
        input_tensor,
        dL_dout_tensor,
        weight_tensor,
        epsilon,
        std::nullopt,
        std::nullopt,
        compute_grad_input,
        compute_packed_partials);
    TT_FATAL(fused_outputs.size() == 2U, "Fused polynorm bw expected 2 outputs, got {}", fused_outputs.size());

    auto& dL_dx = fused_outputs[0];
    if (!compute_packed_partials) {
        return {dL_dx, std::nullopt, std::nullopt};
    }
    auto reduced_partials = ttnn::sum(
        fused_outputs[1].value(),
        /*dim_arg=*/ttsl::SmallVector<int>{0, 1, 2},
        /*keep_dim=*/true,
        /*output_mem_config=*/std::nullopt,
        /*compute_kernel_config=*/core::ComputeKernelConfig::precise());

    // Extract dw0, dw1, dw2, db from the reduced partials on-device (no host roundtrip).
    // The reduced tensor is always (1,1,1,128) = 4 scalar tiles regardless of input size,
    // so the slice/concat cost is constant and negligible.
    constexpr uint32_t tw = tt::constants::TILE_WIDTH;
    std::optional<ttnn::Tensor> dL_dw;
    if (compute_grad_weight) {
        auto dw0 = extract_scalar_at_column(reduced_partials, 0U * tw);
        auto dw1 = extract_scalar_at_column(reduced_partials, 1U * tw);
        auto dw2 = extract_scalar_at_column(reduced_partials, 2U * tw);
        dL_dw = ttnn::concat(std::vector<ttnn::Tensor>{dw0, dw1, dw2}, /*dim=*/3);
    }
    std::optional<ttnn::Tensor> dL_db;
    if (compute_grad_bias) {
        dL_db = extract_scalar_at_column(reduced_partials, 3U * tw);
    }

    return {dL_dx, dL_dw, dL_db};
}

}  // namespace ttml::metal
