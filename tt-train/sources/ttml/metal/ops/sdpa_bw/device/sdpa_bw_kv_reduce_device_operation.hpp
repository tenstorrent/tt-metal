// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "sdpa_bw_kv_device_operation_types.hpp"
#include "sdpa_bw_kv_reduce_program_factory.hpp"

namespace ttml::metal::ops::sdpa_bw::device {

/**
 *   Sums the fp32 per-query-head dK / dV partials of the head-split KV backward over each K/V group
 *   (heads_per_group consecutive heads) into bf16 grad_K (B, kv_heads, S, kE) and grad_V (B, kv_heads, S, vE).
 */
struct SDPABackwardKVReduceDeviceOperation {
    using operation_attributes_t = kv_reduce::operation_attributes_t;
    using tensor_args_t = kv_reduce::tensor_args_t;
    using tensor_return_value_t = kv_reduce::tensor_return_value_t;
    using spec_return_value_t = kv_reduce::spec_return_value_t;
    using program_factory_t = std::variant<SDPABackwardKVReduceProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::sdpa_bw::device

namespace ttnn::prim {

ttml::metal::ops::sdpa_bw::device::SDPABackwardKVReduceDeviceOperation::tensor_return_value_t ttml_sdpa_kv_bw_reduce(
    const ttnn::Tensor& grad_key_partial,
    const ttnn::Tensor& grad_value_partial,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t heads_per_group);

}  // namespace ttnn::prim
