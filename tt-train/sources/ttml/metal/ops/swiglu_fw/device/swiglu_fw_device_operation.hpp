// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"
#include "swiglu_fw_device_operation_types.hpp"
#include "swiglu_fw_program_factory.hpp"

namespace ttml::metal::ops::swiglu_fw::device {

struct SwiGLUForwardDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::swiglu_fw::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::swiglu_fw::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::swiglu_fw::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::swiglu_fw::device::tensor_return_value_t;
    using program_factory_t = std::variant<SwiGLUForwardProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::swiglu_fw::device

namespace ttnn::prim {

ttml::metal::ops::swiglu_fw::device::SwiGLUForwardDeviceOperation::tensor_return_value_t ttml_swiglu_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& m1,
    const ttnn::Tensor& m2,
    const ttnn::Tensor& m3,
    const std::optional<ttnn::Tensor>& preallocated_swiglu = std::nullopt);

}  // namespace ttnn::prim
