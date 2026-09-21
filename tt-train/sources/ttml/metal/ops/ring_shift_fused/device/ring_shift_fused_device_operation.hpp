// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "ring_shift_fused_device_operation_types.hpp"
#include "ring_shift_fused_program_factory.hpp"

namespace ttml::metal::ops::ring_shift_fused {

struct RingShiftFusedDeviceOperation {
    using operation_attributes_t = ring_shift_fused::operation_attributes_t;
    using tensor_args_t = ring_shift_fused::tensor_args_t;
    using spec_return_value_t = ring_shift_fused::spec_return_value_t;
    using tensor_return_value_t = ring_shift_fused::tensor_return_value_t;
    using program_factory_t = std::variant<RingShiftFusedProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::ring_shift_fused

namespace ttnn::prim {

ttml::metal::ops::ring_shift_fused::RingShiftFusedDeviceOperation::tensor_return_value_t ttml_ring_shift_fused(
    const std::vector<ttnn::Tensor>& inputs,
    const std::vector<tt::tt_metal::distributed::MeshSocket>& send_sockets,
    const std::vector<tt::tt_metal::distributed::MeshSocket>& recv_sockets,
    const std::vector<ttnn::Tensor>& preallocated_outputs);

}  // namespace ttnn::prim
