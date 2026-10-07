// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "rmsnorm_bw_device_operation_types.hpp"
#include "rmsnorm_bw_program_factory.hpp"

namespace ttml::metal::ops::rmsnorm_bw::device {

// Phase A: per-(tile-row, slice) partial sums of a * gamma * dL_dout.
struct RMSNormBackwardPartialDeviceOperation {
    using operation_attributes_t = partial::operation_attributes_t;
    using tensor_args_t = partial::tensor_args_t;
    using spec_return_value_t = partial::spec_return_value_t;
    using tensor_return_value_t = partial::tensor_return_value_t;
    using program_factory_t = std::variant<RMSNormBackwardPartialProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// Phase B: dL_da and (optionally) dL_dgamma components.
struct RMSNormBackwardDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::rmsnorm_bw::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::rmsnorm_bw::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::rmsnorm_bw::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::rmsnorm_bw::device::tensor_return_value_t;
    using program_factory_t = std::variant<RMSNormBackwardProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::rmsnorm_bw::device

namespace ttnn::prim {

ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardPartialDeviceOperation::tensor_return_value_t
ttml_rmsnorm_bw_partial(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const ttml::metal::ops::rmsnorm_bw::device::WorkSplit& split);

ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardDeviceOperation::tensor_return_value_t ttml_rmsnorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& rms_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const ttnn::Tensor& partials_tensor,
    const ttml::metal::ops::rmsnorm_bw::device::WorkSplit& split,
    float epsilon = 1e-6F,
    bool compute_dgamma = true);

}  // namespace ttnn::prim
