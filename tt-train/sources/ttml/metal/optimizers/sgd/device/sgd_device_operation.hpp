// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"
#include "sgd_device_operation_types.hpp"
#include "sgd_program_factory.hpp"

namespace ttml::metal::optimizers::sgd::device {

struct SGDDeviceOperation {
    using operation_attributes_t = ttml::metal::optimizers::sgd::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::optimizers::sgd::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::optimizers::sgd::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::optimizers::sgd::device::tensor_return_value_t;
    using program_factory_t = std::variant<SGDProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::optimizers::sgd::device

namespace ttnn::prim {

ttml::metal::optimizers::sgd::device::SGDDeviceOperation::tensor_return_value_t ttml_sgd(
    const ttnn::Tensor& param,
    const ttnn::Tensor& grad,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    const std::optional<ttnn::Tensor>& momentum_buffer = std::nullopt);

}  // namespace ttnn::prim
