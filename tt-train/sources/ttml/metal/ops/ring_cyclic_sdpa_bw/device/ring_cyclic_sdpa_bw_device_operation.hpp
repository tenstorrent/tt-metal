// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/common/const_utils.hpp"
#include "metal/ops/common/ring_sdpa_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ring_cyclic_sdpa_bw_device_operation_types.hpp"
#include "ring_cyclic_sdpa_bw_program_factory.hpp"

namespace ttml::metal::ops::ring_cyclic_sdpa_bw {

struct RingCyclicSDPABackwardDeviceOperation {
    using operation_attributes_t = ring_cyclic_sdpa_bw::operation_attributes_t;
    using tensor_args_t = ring_cyclic_sdpa_bw::tensor_args_t;
    using tensor_return_value_t = ring_cyclic_sdpa_bw::tensor_return_value_t;
    using spec_return_value_t = ring_cyclic_sdpa_bw::spec_return_value_t;
    using program_factory_t = std::variant<RingCyclicSDPABackwardProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::ring_cyclic_sdpa_bw

namespace ttnn::prim {

ttml::metal::ops::ring_cyclic_sdpa_bw::RingCyclicSDPABackwardDeviceOperation::tensor_return_value_t
ttml_ring_cyclic_sdpa_bw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& log_sum_exp,
    const ttnn::Tensor& row_scalar,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type,
    ttml::metal::ops::ring_cyclic_sdpa_bw::RingDirection ring_direction,
    uint32_t rows_per_block_tiles,
    bool use_barrier,
    bool accumulate_into_outputs,
    const std::optional<ttnn::Tensor>& preallocated_grad_query = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_grad_key = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_grad_value = std::nullopt,
    ttml::metal::ops::RingLayout layout = ttml::metal::ops::RingLayout::Contiguous,
    uint32_t zigzag_pair = 0xFFFFFFFFU,
    bool grad_query_in_tile_transposed = false,
    bool grad_query_out_tile_transposed = false);

}  // namespace ttnn::prim
