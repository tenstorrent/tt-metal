// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "depthwise_conv1d_k4_device_operation_types.hpp"
#include "depthwise_conv1d_k4_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::depthwise_conv1d_k4::device {

struct DepthwiseConv1dK4DeviceOperation {
    using operation_attributes_t = ttml::metal::ops::depthwise_conv1d_k4::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::depthwise_conv1d_k4::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::depthwise_conv1d_k4::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::depthwise_conv1d_k4::device::tensor_return_value_t;
    using program_factory_t = std::variant<DepthwiseConv1dK4ProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::depthwise_conv1d_k4::device

namespace ttnn::prim {

ttml::metal::ops::depthwise_conv1d_k4::device::DepthwiseConv1dK4DeviceOperation::tensor_return_value_t
ttml_depthwise_conv1d_k4(
    const ttnn::Tensor& input,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    bool anti_causal,
    const std::optional<ttnn::Tensor>& silu_grad);

}  // namespace ttnn::prim
