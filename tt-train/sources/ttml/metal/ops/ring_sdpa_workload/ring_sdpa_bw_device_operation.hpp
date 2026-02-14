// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ring_sdpa_bw_workload.hpp"
#include "ring_sdpa_device_operation_types.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::ring_sdpa {

// ============== Backward Q Device Operation ==============

struct RingSDPABwQDeviceOperation {
    using operation_attributes_t = RingSDPABwQParams;
    using tensor_args_t = RingSDPABwQInputs;
    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<RingSDPABwQProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);
    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);
};

// ============== Backward KV Device Operation ==============

struct RingSDPABwKVDeviceOperation {
    using operation_attributes_t = RingSDPABwKVParams;
    using tensor_args_t = RingSDPABwKVInputs;
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::tuple<ttnn::Tensor, ttnn::Tensor>;
    using program_factory_t = std::variant<RingSDPABwKVProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);
    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);
};

}  // namespace ttml::metal::ops::ring_sdpa

// Standalone prim functions (like SDPA backward) - avoids register_operation reflection issues
namespace ttml::metal::prim {

ttnn::Tensor ring_sdpa_bw_q(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& intermediates,
    ttnn::Tensor& grad_query,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_sdpa::RingDirection ring_direction);

std::tuple<ttnn::Tensor, ttnn::Tensor> ring_sdpa_bw_kv(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& intermediates,
    ttnn::Tensor& grad_key,
    ttnn::Tensor& grad_value,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_sdpa::RingDirection ring_direction);

}  // namespace ttml::metal::prim
