// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "ring_lse_merge_device_operation_types.hpp"
#include "ring_lse_merge_program_factory.hpp"

namespace ttml::metal::ops::ring_lse_merge::device {

struct RingLseMergeDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::ring_lse_merge::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::ring_lse_merge::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::ring_lse_merge::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::ring_lse_merge::device::tensor_return_value_t;
    using program_factory_t = std::variant<RingLseMergeProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // The outputs are the accumulator and running logsumexp, updated in place: keep their own topologies
    // rather than letting the framework re-derive them from the union of all inputs.
    static std::vector<tt::tt_metal::TensorTopology> compute_output_topologies(
        const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::ring_lse_merge::device

namespace ttnn::prim {

ttml::metal::ops::ring_lse_merge::device::RingLseMergeDeviceOperation::tensor_return_value_t ttml_ring_lse_merge(
    const ttnn::Tensor& output_accum,
    const ttnn::Tensor& running_lse,
    const ttnn::Tensor& step_output,
    const ttnn::Tensor& step_intermediates);

}  // namespace ttnn::prim
