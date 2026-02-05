// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ring_sdpa_device_operation_types.hpp"
#include "ring_sdpa_workload.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::ring_sdpa {

struct RingSDPADeviceOperation {
    using operation_attributes_t = RingSDPAParams;
    using tensor_args_t = RingSDPAInputs;
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::tuple<ttnn::Tensor, ttnn::Tensor>;  // output, intermediates
    using program_factory_t = std::variant<RingSDPAProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

}  // namespace ttml::metal::ops::ring_sdpa

// Standalone prim function (like SDPA backward) - avoids register_operation reflection issues
namespace ttml::metal::prim {

std::tuple<ttnn::Tensor, ttnn::Tensor> ring_sdpa(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    ttnn::Tensor& output,
    ttnn::Tensor& intermediates,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_sdpa::RingDirection ring_direction = ttml::metal::ops::ring_sdpa::RingDirection::Backward);

}  // namespace ttml::metal::prim
