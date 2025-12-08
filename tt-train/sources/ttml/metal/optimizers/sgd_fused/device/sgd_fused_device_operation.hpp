// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"
#include "sgd_fused_device_operation_types.hpp"
#include "sgd_fused_program_factory.hpp"

namespace ttml::metal::optimizers::sgd_fused::device {

struct SGDFusedDeviceOperation {
    using operation_attributes_t = operation_attributes_t;
    using tensor_args_t = tensor_args_t;
    using spec_return_value_t = spec_return_value_t;
    using tensor_return_value_t = tensor_return_value_t;
    using program_factory_t = std::variant<SGDFusedProgramFactory>;

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
        float lr,
        float momentum,
        float dampening,
        float weight_decay,
        bool nesterov,
        const std::optional<ttnn::Tensor>& momentum_buffer = std::nullopt);
};

}  // namespace ttml::metal::optimizers::sgd_fused::device

namespace ttnn::prim {

constexpr auto ttml_sgd_fused = ttnn::register_operation<
    "ttnn::prim::ttml_sgd_fused",
    ttml::metal::optimizers::sgd_fused::device::SGDFusedDeviceOperation>();
}  // namespace ttnn::prim
