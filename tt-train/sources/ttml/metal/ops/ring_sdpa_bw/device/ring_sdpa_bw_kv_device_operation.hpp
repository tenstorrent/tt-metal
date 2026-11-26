// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "ring_sdpa_bw_kv_device_operation_types.hpp"
#include "ring_sdpa_bw_kv_program_factory.hpp"

namespace ttml::metal::ops::ring_sdpa_bw {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

}  // namespace ttml::metal::ops::ring_sdpa_bw

namespace ttml::metal::ops::ring_sdpa_bw::kv {

struct RingSDPABwKVDeviceOperation {
    using operation_attributes_t = kv::operation_attributes_t;
    using tensor_args_t = kv::tensor_args_t;
    using tensor_return_value_t = kv::tensor_return_value_t;
    using spec_return_value_t = kv::spec_return_value_t;
    using program_factory_t = std::variant<RingSDPABwKVProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::ring_sdpa_bw::kv

namespace ttnn::prim {

ttml::metal::ops::ring_sdpa_bw::kv::RingSDPABwKVDeviceOperation::tensor_return_value_t ttml_ring_sdpa_bw_kv(
    std::vector<ttml::metal::ops::ring_sdpa_bw::kv::RingSDPABwKVChunk> chunks,
    const ttnn::Tensor& u_scaler,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_sdpa_bw::RingDirection ring_direction =
        ttml::metal::ops::ring_sdpa_bw::RingDirection::Backward,
    ttml::metal::ops::RingLoadBalancing load_balancing = ttml::metal::ops::RingLoadBalancing::None);

}  // namespace ttnn::prim
