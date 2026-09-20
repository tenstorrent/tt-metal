// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "ring_cyclic_sdpa_fw_device_operation_types.hpp"
#include "ring_cyclic_sdpa_fw_program_factory.hpp"

namespace ttml::metal::ops::ring_cyclic_sdpa_fw {

struct RingCyclicSdpaFwDeviceOperation {
    using operation_attributes_t = ring_cyclic_sdpa_fw::operation_attributes_t;
    using tensor_args_t = ring_cyclic_sdpa_fw::tensor_args_t;
    using spec_return_value_t = ring_cyclic_sdpa_fw::spec_return_value_t;
    using tensor_return_value_t = ring_cyclic_sdpa_fw::tensor_return_value_t;
    using program_factory_t = std::variant<RingCyclicSdpaFwProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::ring_cyclic_sdpa_fw

namespace ttnn::prim {

ttml::metal::ops::ring_cyclic_sdpa_fw::RingCyclicSdpaFwDeviceOperation::tensor_return_value_t
ttml_ring_cyclic_sdpa_fw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_cyclic_sdpa_fw::RingDirection ring_direction,
    bool zigzag,
    ttml::metal::ops::ZigzagVisitor visitor,
    uint32_t rows_per_block_tiles,
    const std::optional<ttnn::Tensor>& preallocated_output,
    const std::optional<ttnn::Tensor>& preallocated_intermediates);

}  // namespace ttnn::prim
