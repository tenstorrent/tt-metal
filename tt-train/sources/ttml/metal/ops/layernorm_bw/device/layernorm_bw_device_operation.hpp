// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "layernorm_bw_device_operation_types.hpp"
#include "layernorm_bw_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::layernorm_bw::device {

struct LayerNormBackwardDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::layernorm_bw::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::layernorm_bw::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::layernorm_bw::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::layernorm_bw::device::tensor_return_value_t;
    using program_factory_t = std::variant<LayerNormBackwardProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::layernorm_bw::device

namespace ttnn::prim {

ttml::metal::ops::layernorm_bw::device::LayerNormBackwardDeviceOperation::tensor_return_value_t ttml_layernorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& mean_tensor,
    const ttnn::Tensor& rstd_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const std::optional<ttnn::Tensor>& preallocated_dx = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_dgamma_components = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_dbeta_components = std::nullopt);

}  // namespace ttnn::prim
