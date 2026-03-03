// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"
#include "swiglu_grad_device_operation_types.hpp"
#include "swiglu_grad_program_factory.hpp"

namespace ttml::metal::ops::swiglu_grad::device {

struct SwiGLUGradDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::swiglu_grad::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::swiglu_grad::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::swiglu_grad::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::swiglu_grad::device::tensor_return_value_t;
    using program_factory_t = std::variant<SwiGLUGradProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::swiglu_grad::device

namespace ttnn::prim {

ttml::metal::ops::swiglu_grad::device::SwiGLUGradDeviceOperation::tensor_return_value_t ttml_swiglu_grad(
    const ttnn::Tensor& linear1,
    const ttnn::Tensor& gate,
    const ttnn::Tensor& dL_dprod,
    const std::optional<ttnn::Tensor>& preallocated_dL_dlinear1 = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_dL_dgate = std::nullopt);

}  // namespace ttnn::prim
