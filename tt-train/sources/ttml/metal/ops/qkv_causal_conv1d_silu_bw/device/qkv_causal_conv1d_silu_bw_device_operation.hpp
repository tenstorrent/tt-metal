// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "metal/ttnn_all_includes.hpp"
#include "qkv_causal_conv1d_silu_bw_device_operation_types.hpp"
#include "qkv_causal_conv1d_silu_bw_program_factory.hpp"

namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device {

// Outputs: grad_input [1,T,Q+K+V] TILE bf16 and, with compute_tap_grads, grad_tap_partial [4,splits*32,Q+K+V]
// TILE fp32 (host sums the rows, then casts to bf16). Rows of one tap stack per split so one row reduction
// finishes the sum.
struct QkvCausalConv1dSiluBackwardOperation {
    using operation_attributes_t = QkvCausalConv1dSiluBackwardParams;
    using tensor_args_t = QkvCausalConv1dSiluBackwardInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<QkvCausalConv1dSiluBackwardProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::qkv_causal_conv1d_silu_bw::device

namespace ttnn::prim {

std::vector<ttnn::Tensor> ttml_qkv_causal_conv1d_silu_bw(
    const ttnn::Tensor& grad_q,
    const ttnn::Tensor& grad_k,
    const ttnn::Tensor& grad_v,
    const ttnn::Tensor& input,
    const ttnn::Tensor& history,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    const ttml::metal::ops::qkv_causal_conv1d_silu_bw::device::QkvCausalConv1dSiluBackwardParams& params,
    const std::optional<ttnn::Tensor>& preallocated_grad_input = std::nullopt);

}  // namespace ttnn::prim
