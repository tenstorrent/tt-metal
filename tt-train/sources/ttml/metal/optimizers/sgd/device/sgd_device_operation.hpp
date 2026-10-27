// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
    // The update writes `param` in place and hands it back as the op output. Pin the output's topology to the
    // parameter's own: without this the framework re-derives it from the union of all inputs, so a gradient carrying
    // a stale label (a CCL output that kept a Shard on a reduced axis, say) would relabel the parameter, and the
    // checkpointer gathers by that label.
    static std::vector<tt::tt_metal::TensorTopology> compute_output_topologies(
        const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::optimizers::sgd::device

namespace ttnn::prim {

ttml::metal::optimizers::sgd::device::SGDDeviceOperation::tensor_return_value_t sgd(
    const ttnn::Tensor& param,
    const ttnn::Tensor& grad,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    const std::optional<ttnn::Tensor>& momentum_buffer = std::nullopt);

}  // namespace ttnn::prim
