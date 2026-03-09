// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "adamw_device_operation_types.hpp"
#include "adamw_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::optimizers::adamw::device {

struct AdamWDeviceOperation {
    using operation_attributes_t = ttml::metal::optimizers::adamw::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::optimizers::adamw::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::optimizers::adamw::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::optimizers::adamw::device::tensor_return_value_t;
    using program_factory_t = std::variant<AdamWProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::optimizers::adamw::device

namespace ttnn::prim {

ttml::metal::optimizers::adamw::device::AdamWDeviceOperation::tensor_return_value_t adamw(
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
    ttml::metal::StochasticRounding stochastic_rounding);

}  // namespace ttnn::prim
