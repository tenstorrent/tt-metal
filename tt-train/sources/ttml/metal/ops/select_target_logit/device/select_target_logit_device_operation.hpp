// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"
#include "select_target_logit_device_operation_types.hpp"
#include "select_target_logit_program_factory.hpp"

namespace ttml::metal::ops::select_target_logit::device {

struct SelectTargetLogitDeviceOperation {
    using operation_attributes_t = ttml::metal::ops::select_target_logit::device::operation_attributes_t;
    using tensor_args_t = ttml::metal::ops::select_target_logit::device::tensor_args_t;
    using spec_return_value_t = ttml::metal::ops::select_target_logit::device::spec_return_value_t;
    using tensor_return_value_t = ttml::metal::ops::select_target_logit::device::tensor_return_value_t;
    using program_factory_t = std::variant<SelectTargetLogitProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::select_target_logit::device

namespace ttnn::prim {

// Selects the target-logit value at each (n, s) position from a (possibly vocab-sharded) logit
// tensor.  The shard window for each device is derived inside the program factory:
//
//   tp_rank        = cluster_axis ? mesh_coord[*cluster_axis] : 0
//   device_first_v = first_v + tp_rank * local_V
//   device_last_v  = device_first_v + local_V
//
// Real callers (e.g. vocab-parallel cross-entropy loss) pass `local_V` and the TP `cluster_axis`
// and leave `first_v = 0`.  `first_v` exists so single-device unit tests can still simulate
// non-zero shard windows without standing up a multi-device mesh.
ttml::metal::ops::select_target_logit::device::SelectTargetLogitDeviceOperation::tensor_return_value_t
ttml_select_target_logit(
    const ttnn::Tensor& logit,
    const ttnn::Tensor& target,
    uint32_t local_V,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    uint32_t first_v = 0U,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim
