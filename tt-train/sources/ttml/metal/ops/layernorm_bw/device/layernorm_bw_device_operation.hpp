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
    using operation_attributes_t = operation_attributes_t;
    using tensor_args_t = tensor_args_t;
    using spec_return_value_t = spec_return_value_t;
    using tensor_return_value_t = tensor_return_value_t;
    using program_factory_t = std::variant<LayerNormBackwardProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& gamma_tensor,
        const ttnn::Tensor& mean_tensor,
        const ttnn::Tensor& rstd_tensor,
        const ttnn::Tensor& dL_dout_tensor,
        float epsilon = 1e-5F,
        const std::optional<ttnn::Tensor>& preallocated_dx = std::nullopt,
        const std::optional<ttnn::Tensor>& preallocated_dgamma_components = std::nullopt,
        const std::optional<ttnn::Tensor>& preallocated_dbeta_components = std::nullopt);
};

}  // namespace ttml::metal::ops::layernorm_bw::device

namespace ttnn::prim {
constexpr auto ttml_layernorm_bw = ttnn::register_operation<
    "ttnn::prim::ttml_layernorm_bw",
    ttml::metal::ops::layernorm_bw::device::LayerNormBackwardDeviceOperation>();
}
