// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "layernorm_fw_device_operation_types.hpp"
#include "layernorm_fw_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::layernorm_fw::device {

struct LayerNormForwardDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::layernorm_fw::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::layernorm_fw::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::layernorm_fw::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::layernorm_fw::device::tensor_return_value_t;
    using program_factory_t = std::variant<LayerNormForwardProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::layernorm_fw::device

namespace ttnn::prim {

ttml::metal::ops::layernorm_fw::device::LayerNormForwardDeviceOperation::tensor_return_value_t ttml_layernorm_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& beta_tensor,
    float epsilon = 1e-5F,
    bool return_mean_rstd = false,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_mean = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_rstd = std::nullopt);

}  // namespace ttnn::prim
