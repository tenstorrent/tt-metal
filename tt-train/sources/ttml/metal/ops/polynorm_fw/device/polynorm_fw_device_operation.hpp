// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"
#include "polynorm_fw_device_operation_types.hpp"
#include "polynorm_fw_program_factory.hpp"

namespace ttml::metal::ops::polynorm_fw::device {

struct PolyNormForwardDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::polynorm_fw::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::polynorm_fw::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::polynorm_fw::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::polynorm_fw::device::tensor_return_value_t;
    using program_factory_t = std::variant<PolyNormForwardProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::polynorm_fw::device

namespace ttnn::prim {

ttml::metal::ops::polynorm_fw::device::PolyNormForwardDeviceOperation::tensor_return_value_t ttml_polynorm_fw(
    const ttnn::Tensor& input_tensor,
    float w0,
    float w1,
    float w2,
    float bias,
    float epsilon = 1e-5F,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim
