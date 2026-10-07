// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gated_rmsnorm_device_operation_types.hpp"
#include "gated_rmsnorm_fw_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gated_rmsnorm::device {

struct GatedRmsNormForwardDeviceOperation {
    using operation_attributes_t = fw::operation_attributes_t;
    using tensor_args_t = fw::tensor_args_t;
    using spec_return_value_t = fw::spec_return_value_t;
    using tensor_return_value_t = fw::tensor_return_value_t;
    using program_factory_t = std::variant<GatedRmsNormForwardProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::gated_rmsnorm::device

namespace ttnn::prim {

ttml::metal::ops::gated_rmsnorm::device::GatedRmsNormForwardDeviceOperation::tensor_return_value_t
ttml_gated_rmsnorm_fw(const ttnn::Tensor& input, const ttnn::Tensor& gate, const ttnn::Tensor& gamma, float epsilon);

}  // namespace ttnn::prim
