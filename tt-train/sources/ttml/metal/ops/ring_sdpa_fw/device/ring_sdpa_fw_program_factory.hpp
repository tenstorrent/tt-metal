// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ops/sdpa_fw/device/sdpa_fw_program_factory.hpp"
#include "metal/ops/sdpa_fw/sdpa_fw_program_config.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ring_sdpa_fw_device_operation_types.hpp"

namespace ttml::metal::ops::ring_sdpa_fw {

// The program main ran for a ring shard of St sequence tiles, so ring SDPA is unchanged by the chunk knobs
// and the cost-based scheduler: one query tile-row per item, the largest of four, two or one K/V tiles per
// step that divides St (main's pick_sk_chunk_t; a shard can be a tile or two long), no head stacking, and
// the causal items placed as heavy/light pairs, never cut. Its only device test is the Galaxy ring test.
SDPAForwardProgramConfig ring_program_config(uint32_t St);

// Forward Program Factory: one SDPA forward program per device, so the SDPA shared variables are reused.
struct RingSDPAFwProgramFactory {
    using shared_variables_t = sdpa_fw::device::SDPAForwardProgramFactory::shared_variables_t;
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

}  // namespace ttml::metal::ops::ring_sdpa_fw
