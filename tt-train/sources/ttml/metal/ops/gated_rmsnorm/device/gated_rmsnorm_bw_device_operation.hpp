// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gated_rmsnorm_bw_program_factory.hpp"
#include "gated_rmsnorm_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gated_rmsnorm::device {

struct GatedRmsNormBackwardDeviceOperation {
    using operation_attributes_t = bw::operation_attributes_t;
    using tensor_args_t = bw::tensor_args_t;
    using spec_return_value_t = bw::spec_return_value_t;
    using tensor_return_value_t = bw::tensor_return_value_t;
    using program_factory_t = std::variant<GatedRmsNormBackwardProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::gated_rmsnorm::device

namespace ttnn::prim {

ttml::metal::ops::gated_rmsnorm::device::GatedRmsNormBackwardDeviceOperation::tensor_return_value_t
ttml_gated_rmsnorm_bw(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& gamma,
    const ttnn::Tensor& dL_dout,
    float epsilon,
    bool compute_dgamma);

}  // namespace ttnn::prim
