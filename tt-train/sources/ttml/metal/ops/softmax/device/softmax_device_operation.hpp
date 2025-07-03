// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"
#include "softmax_device_operation_types.hpp"
#include "softmax_program_factory.hpp"

namespace ttml::metal::ops::softmax::device {

struct SoftmaxDeviceOperation {
    using operation_attributes_t = operation_attributes_t;
    using tensor_args_t = tensor_args_t;
    using spec_return_value_t = spec_return_value_t;
    using tensor_return_value_t = tensor_return_value_t;
    using program_factory_t = std::variant<SoftmaxProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& input_tensor,
        int32_t dim = 3U,  // Use last dimension by default
        const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt);
};

}  // namespace ttml::metal::ops::softmax::device

namespace ttnn::prim {

constexpr auto ttml_softmax =
    ttnn::register_operation<"ttnn::prim::ttml_softmax", ttml::metal::ops::softmax::device::SoftmaxDeviceOperation>();
}  // namespace ttnn::prim
