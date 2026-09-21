// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "metal/ttnn_all_includes.hpp"
#include "ring_shift_fused_device_operation_types.hpp"

namespace ttml::metal::ops::ring_shift_fused {

struct RingShiftFusedProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel{};
        tt::tt_metal::KernelHandle writer_kernel{};
        tt::tt_metal::KernelHandle receiver_kernel{};
        std::vector<tt::tt_metal::CoreCoord> sender_cores;
        std::vector<tt::tt_metal::CoreCoord> receiver_cores;
        uint32_t num_tensors{};
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttml::metal::ops::ring_shift_fused
