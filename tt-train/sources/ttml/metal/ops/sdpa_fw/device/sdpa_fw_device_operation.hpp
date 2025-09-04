// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "sdpa_fw_device_operation_types.hpp"
#include "sdpa_fw_program_factory.hpp"

namespace ttml::metal::ops::sdpa_fw::device {

struct SDPAForwardDeviceOperation {
    using operation_attributes_t = operation_attributes_t;
    using tensor_args_t = tensor_args_t;
    using tensor_return_value_t = tensor_return_value_t;
    using spec_return_value_t = spec_return_value_t;
    using program_factory_t = std::variant<SDPAForwardProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& query_tensor,
        const ttnn::Tensor& key_tensor,
        const ttnn::Tensor& value_tensor,
        const std::optional<ttnn::Tensor>& mask,  // attention mask
        const uint32_t q_heads,                   // num of query heads
        const uint32_t kv_heads,                  // num of key/value heads
        const float dropout_probability = 0.8F,   // default value
        const bool return_intermediates = false,
        const std::optional<ttnn::Tensor>& preallocated_intermediate = std::nullopt,
        const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt);
};

}  // namespace ttml::metal::ops::sdpa_fw::device

namespace ttnn::prim {

constexpr auto ttml_sdpa_fw = ttnn::
    register_operation<"ttnn::prim::ttml_sdpa_fw", ttml::metal::ops::sdpa_fw::device::SDPAForwardDeviceOperation>();

}  // namespace ttnn::prim
