// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "ring_sdpa_bw_q_device_operation_types.hpp"
#include "ring_sdpa_bw_q_program_factory.hpp"

namespace ttml::metal::ops::ring_sdpa_bw {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

}  // namespace ttml::metal::ops::ring_sdpa_bw

namespace ttml::metal::ops::ring_sdpa_bw::q {

struct RingSDPABwQDeviceOperation {
    using operation_attributes_t = q::operation_attributes_t;
    using tensor_args_t = q::tensor_args_t;
    using tensor_return_value_t = q::tensor_return_value_t;
    using spec_return_value_t = q::spec_return_value_t;
    using program_factory_t = std::variant<RingSDPABwQProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::ring_sdpa_bw::q

namespace ttnn::prim {

ttml::metal::ops::ring_sdpa_bw::q::RingSDPABwQDeviceOperation::tensor_return_value_t ttml_ring_sdpa_bw_q(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& intermediates,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::None,
    ttml::metal::ops::ring_sdpa_bw::RingDirection ring_direction =
        ttml::metal::ops::ring_sdpa_bw::RingDirection::Backward,
    const std::optional<ttnn::Tensor>& preallocated_grad_query = std::nullopt);

}  // namespace ttnn::prim
