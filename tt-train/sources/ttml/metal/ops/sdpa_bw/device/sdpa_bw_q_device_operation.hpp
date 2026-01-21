// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "sdpa_bw_q_device_operation_types.hpp"
#include "sdpa_bw_q_program_factory.hpp"

namespace ttml::metal::ops::sdpa_bw::device {

struct SDPABackwardQDeviceOperation {
    using operation_attributes_t = q::operation_attributes_t;
    using tensor_args_t = q::tensor_args_t;
    using tensor_return_value_t = q::tensor_return_value_t;
    using spec_return_value_t = q::spec_return_value_t;
    using program_factory_t = std::variant<SDPABackwardQProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::sdpa_bw::device

namespace ttnn::prim {

ttml::metal::ops::sdpa_bw::device::SDPABackwardQDeviceOperation::tensor_return_value_t ttml_sdpa_q_bw(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query_tensor,
    const ttnn::Tensor& key_tensor,
    const ttnn::Tensor& value_tensor,
    const std::optional<ttnn::Tensor>& mask,
    const ttnn::Tensor& intermediates,
    const float dropout_probability = 0.0F,
    const bool fp32_dest_acc_en = true,
    const std::optional<ttnn::Tensor>& preallocated_grad_query = std::nullopt);

}  // namespace ttnn::prim
