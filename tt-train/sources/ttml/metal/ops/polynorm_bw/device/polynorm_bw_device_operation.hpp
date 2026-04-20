// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "polynorm_bw_device_operation_types.hpp"
#include "polynorm_bw_program_factory.hpp"

namespace ttml::metal::ops::polynorm3_bw::device {

struct PolyNorm3BackwardDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::polynorm3_bw::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::polynorm3_bw::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::polynorm3_bw::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::polynorm3_bw::device::tensor_return_value_t;
    using program_factory_t = std::variant<PolyNorm3BackwardProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::polynorm3_bw::device

namespace ttnn::prim {

ttml::metal::ops::polynorm3_bw::device::PolyNorm3BackwardDeviceOperation::tensor_return_value_t ttml_polynorm3_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const ttnn::Tensor& weight_tensor,
    float epsilon = 1e-5F,
    const std::optional<ttnn::Tensor>& preallocated_dL_dx = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_packed_partials = std::nullopt);

}  // namespace ttnn::prim
