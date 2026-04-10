// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "gather_logit_device_operation_types.hpp"
#include "gather_logit_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gather_logit::device {

struct GatherLogitDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::gather_logit::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::gather_logit::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::gather_logit::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::gather_logit::device::tensor_return_value_t;
    using program_factory_t = std::variant<GatherLogitProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::gather_logit::device

namespace ttnn::prim {

ttml::metal::ops::gather_logit::device::GatherLogitDeviceOperation::tensor_return_value_t ttml_gather_logit(
    const ttnn::Tensor& logit,
    const ttnn::Tensor& target,
    uint32_t first_v,
    uint32_t last_v,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim
