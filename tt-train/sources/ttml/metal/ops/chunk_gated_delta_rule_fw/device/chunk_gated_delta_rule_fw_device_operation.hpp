// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "chunk_gated_delta_rule_fw_device_operation_types.hpp"
#include "chunk_gated_delta_rule_fw_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::chunk_gated_delta_rule_fw::device {

struct ChunkGatedDeltaRuleForwardPrepDeviceOperation {
    using operation_attributes_t = ChunkGatedDeltaRuleForwardPrepParams;
    using tensor_args_t = ChunkGatedDeltaRuleForwardPrepInputs;
    using spec_return_value_t = ttml::metal::ops::chunk_gated_delta_rule_fw::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::chunk_gated_delta_rule_fw::device::tensor_return_value_t;
    using program_factory_t = std::variant<ChunkGatedDeltaRuleForwardPrepProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

struct ChunkGatedDeltaRuleForwardScanDeviceOperation {
    using operation_attributes_t = ChunkGatedDeltaRuleForwardScanParams;
    using tensor_args_t = ChunkGatedDeltaRuleForwardScanInputs;
    using spec_return_value_t = ttml::metal::ops::chunk_gated_delta_rule_fw::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::chunk_gated_delta_rule_fw::device::tensor_return_value_t;
    using program_factory_t = std::variant<ChunkGatedDeltaRuleForwardScanProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::chunk_gated_delta_rule_fw::device

namespace ttnn::prim {

ttml::metal::ops::chunk_gated_delta_rule_fw::device::tensor_return_value_t ttml_chunk_gated_delta_rule_fw_prep(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    uint32_t chunk_size,
    uint32_t key_head_dim,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const tt::tt_metal::MemoryConfig& output_memory_config);

// `prep` is the output of ttml_chunk_gated_delta_rule_fw_prep, indexed by PrepOutputIndex.
ttml::metal::ops::chunk_gated_delta_rule_fw::device::tensor_return_value_t ttml_chunk_gated_delta_rule_fw_scan(
    const ttml::metal::ops::chunk_gated_delta_rule_fw::device::tensor_return_value_t& prep,
    const std::optional<ttnn::Tensor>& initial_state,
    uint32_t num_value_heads,
    bool output_head_major,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const tt::tt_metal::MemoryConfig& output_memory_config);

}  // namespace ttnn::prim
