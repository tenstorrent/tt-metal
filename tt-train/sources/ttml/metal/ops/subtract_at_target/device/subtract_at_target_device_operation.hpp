// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"
#include "subtract_at_target_device_operation_types.hpp"
#include "subtract_at_target_program_factory.hpp"

namespace ttml::metal::ops::subtract_at_target::device {

struct SubtractAtTargetDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::subtract_at_target::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::subtract_at_target::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::subtract_at_target::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::subtract_at_target::device::tensor_return_value_t;
    using program_factory_t = std::variant<SubtractAtTargetProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::subtract_at_target::device

namespace ttnn::prim {

ttml::metal::ops::subtract_at_target::device::SubtractAtTargetDeviceOperation::tensor_return_value_t
ttml_subtract_at_target(
    const ttnn::Tensor& input,
    const ttnn::Tensor& target,
    uint32_t first_v,
    uint32_t last_v,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt,
    float subtract_value = 1.0F);

}  // namespace ttnn::prim
