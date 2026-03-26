// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"
#include "swiglu_elemwise_bw_device_operation_types.hpp"
#include "swiglu_elemwise_bw_program_factory.hpp"

namespace ttml::metal::ops::swiglu_elemwise_bw::device {

struct SwigluElemwiseBwDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::swiglu_elemwise_bw::device::SwigluElemwiseBwParams;
    using tensor_args_t = ttml::metal::ops::swiglu_elemwise_bw::device::SwigluElemwiseBwInputs;
    using spec_return_value_t = ttml::metal::ops::swiglu_elemwise_bw::device::SwigluElemwiseBwOutputSpecs;
    using tensor_return_value_t = ttml::metal::ops::swiglu_elemwise_bw::device::SwigluElemwiseBwResult;
    using program_factory_t = std::variant<SwigluElemwiseBwProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::swiglu_elemwise_bw::device

namespace ttnn::prim {

ttml::metal::ops::swiglu_elemwise_bw::device::SwigluElemwiseBwDeviceOperation::tensor_return_value_t
ttml_swiglu_elemwise_bw(
    const ttnn::Tensor& linear1,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& dL_dprod,
    const std::optional<ttnn::Tensor>& preallocated_dL_dlinear1 = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_dL_dgate = std::nullopt);

}  // namespace ttnn::prim
