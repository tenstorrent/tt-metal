// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ops/cyclic_sdpa_bw/device/cyclic_sdpa_bw_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ring_cyclic_sdpa_bw_device_operation_types.hpp"

namespace ttml::metal::ops::ring_cyclic_sdpa_bw {

struct RingCyclicSDPABackwardProgramFactory {
    // One cyclic program per participating chip, so the shared variables are
    // the cyclic factory's own.
    using shared_variables_t = cyclic_sdpa_bw::device::CyclicSDPABackwardProgramFactory::shared_variables_t;
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

}  // namespace ttml::metal::ops::ring_cyclic_sdpa_bw
