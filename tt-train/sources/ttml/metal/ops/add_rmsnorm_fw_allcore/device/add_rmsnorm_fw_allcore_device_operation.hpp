// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "add_rmsnorm_fw_allcore_device_operation_types.hpp"
#include "add_rmsnorm_fw_allcore_program_factory.hpp"
#include "metal/ops/rmsnorm_bw_allcore/device/rmsnorm_bw_allcore_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::add_rmsnorm_fw_allcore::device {

// Phase A: sum = residual + delta and per-(tile-row, slice) partial sums of sum^2.
struct AddRMSNormForwardPartialDeviceOperation {
    using operation_attributes_t = partial::operation_attributes_t;
    using tensor_args_t = partial::tensor_args_t;
    using spec_return_value_t = partial::spec_return_value_t;
    using tensor_return_value_t = partial::tensor_return_value_t;
    using program_factory_t = std::variant<AddRMSNormForwardPartialProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// Phase B: rms and out = sum * gamma / rms.
struct AddRMSNormForwardDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::add_rmsnorm_fw_allcore::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::add_rmsnorm_fw_allcore::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::add_rmsnorm_fw_allcore::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::add_rmsnorm_fw_allcore::device::tensor_return_value_t;
    using program_factory_t = std::variant<AddRMSNormForwardProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::add_rmsnorm_fw_allcore::device

namespace ttnn::prim {

ttml::metal::ops::add_rmsnorm_fw_allcore::device::AddRMSNormForwardPartialDeviceOperation::tensor_return_value_t
ttml_add_rmsnorm_fw_allcore_partial(
    const ttnn::Tensor& residual_tensor,
    const ttnn::Tensor& delta_tensor,
    const ttml::metal::ops::rmsnorm_bw_allcore::device::WorkSplit& split);

ttml::metal::ops::add_rmsnorm_fw_allcore::device::AddRMSNormForwardDeviceOperation::tensor_return_value_t
ttml_add_rmsnorm_fw_allcore(
    const ttnn::Tensor& sum_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& partials_tensor,
    const ttml::metal::ops::rmsnorm_bw_allcore::device::WorkSplit& split,
    float epsilon);

}  // namespace ttnn::prim
