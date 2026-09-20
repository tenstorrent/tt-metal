// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ring_ttnn_sdpa_fw_device_operation_types.hpp"

namespace ttml::metal::ops::ring_ttnn_sdpa_fw {

// Nothing to keep per program: ttnn's SDPA is a ProgramDescriptor-style
// operation, so a cache hit rebuilds the descriptor from the current tensors
// and applies its runtime arguments to the cached program.
struct RingTtnnSdpaFwSharedVariables {};

struct RingTtnnSdpaFwProgramFactory {
    using shared_variables_t = RingTtnnSdpaFwSharedVariables;
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

}  // namespace ttml::metal::ops::ring_ttnn_sdpa_fw
