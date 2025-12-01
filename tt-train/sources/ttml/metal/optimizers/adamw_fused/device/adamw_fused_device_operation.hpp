// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "adamw_fused_device_operation_types.hpp"
#include "adamw_fused_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::optimizers::adamw_fused::device {

struct AdamWFusedDeviceOperation {
    using operation_attributes_t = operation_attributes_t;
    using tensor_args_t = tensor_args_t;
    using spec_return_value_t = spec_return_value_t;
    using tensor_return_value_t = tensor_return_value_t;
    using program_factory_t = std::variant<AdamWFusedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& param,
        const ttnn::Tensor& grad,
        const ttnn::Tensor& exp_avg,
        const ttnn::Tensor& exp_avg_sq,
        const std::optional<ttnn::Tensor>& max_exp_avg_sq,
        float lr,
        float beta1,
        float beta2,
        float beta1_pow,
        float beta2_pow,
        float epsilon,
        float weight_decay,
        bool amsgrad,
        bool stochastic_rounding,
        uint32_t step);
};

}  // namespace ttml::metal::optimizers::adamw_fused::device

namespace ttnn::prim {

constexpr auto ttml_adamw_fused = ttnn::register_operation<
    "ttnn::prim::ttml_adamw_fused",
    ttml::metal::optimizers::adamw_fused::device::AdamWFusedDeviceOperation>();
}  // namespace ttnn::prim
