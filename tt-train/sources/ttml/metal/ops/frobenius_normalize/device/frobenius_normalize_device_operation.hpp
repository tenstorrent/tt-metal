// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"
#include "frobenius_normalize_device_operation_types.hpp"
#include "frobenius_normalize_program_factory.hpp"

namespace ttml::metal::ops::frobenius_normalize::device {

struct FrobeniusNormalizeDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::frobenius_normalize::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::frobenius_normalize::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::frobenius_normalize::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::frobenius_normalize::device::tensor_return_value_t;
    using program_factory_t = std::variant<FrobeniusNormalizeProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::frobenius_normalize::device

namespace ttnn::prim {

ttml::metal::ops::frobenius_normalize::device::FrobeniusNormalizeDeviceOperation::tensor_return_value_t
ttml_frobenius_normalize(
    const ttnn::Tensor& input_tensor,
    float epsilon = 1e-7F,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim
