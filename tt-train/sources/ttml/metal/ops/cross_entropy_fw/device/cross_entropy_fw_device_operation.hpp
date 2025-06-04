// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "cross_entropy_fw_device_operation_types.hpp"
#include "cross_entropy_fw_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cross_entropy_fw::device {

struct CrossEntropyForwardDeviceOperation {
    using operation_attributes_t = operation_attributes_t;
    using tensor_args_t = tensor_args_t;
    using spec_return_value_t = spec_return_value_t;
    using tensor_return_value_t = tensor_return_value_t;
    using program_factory_t = std::variant<CrossEntropyForwardProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& target_tensor,
        const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt);
};

}  // namespace ttml::metal::ops::cross_entropy_fw::device

namespace ttnn::prim {

constexpr auto ttml_cross_entropy_fw = ttnn::register_operation<
    "ttnn::prim::ttml_cross_entropy_fw",
    ttml::metal::ops::cross_entropy_fw::device::CrossEntropyForwardDeviceOperation>();
}  // namespace ttnn::prim
