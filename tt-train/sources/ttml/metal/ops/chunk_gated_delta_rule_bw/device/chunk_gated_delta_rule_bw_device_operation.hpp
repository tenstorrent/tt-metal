// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "chunk_gated_delta_rule_bw_device_operation_types.hpp"
#include "chunk_gated_delta_rule_bw_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::chunk_gated_delta_rule_bw::device {

struct ChunkGatedDeltaRuleBackwardDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::chunk_gated_delta_rule_bw::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::chunk_gated_delta_rule_bw::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::chunk_gated_delta_rule_bw::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::chunk_gated_delta_rule_bw::device::tensor_return_value_t;
    using program_factory_t = std::variant<ChunkGatedDeltaRuleBackwardProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::chunk_gated_delta_rule_bw::device

namespace ttnn::prim {

ttml::metal::ops::chunk_gated_delta_rule_bw::device::ChunkGatedDeltaRuleBackwardDeviceOperation::tensor_return_value_t
ttml_chunk_gated_delta_rule_bw(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& d_out,
    const std::optional<ttnn::Tensor>& initial_state,
    const std::optional<ttnn::Tensor>& d_final_state,
    const std::optional<ttnn::Tensor>& h,
    const std::optional<ttnn::Tensor>& v_new,
    const std::optional<ttnn::Tensor>& a,
    uint32_t chunk_size,
    uint32_t key_head_dim,
    float scale,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const tt::tt_metal::MemoryConfig& output_memory_config);

}  // namespace ttnn::prim
