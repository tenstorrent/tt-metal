// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ring_sdpa_fw_device_operation_types.hpp"

namespace ttml::metal::ops::ring_sdpa_fw {

// Forward Program Factory
struct RingSDPAFwSharedVariables {
    // SDPA kernel handles for runtime argument updates
    tt::tt_metal::KernelHandle sdpa_fw_reader_kernel{};
    tt::tt_metal::KernelHandle sdpa_fw_writer_kernel{};
    tt::tt_metal::KernelHandle sdpa_fw_kernel_group_1{};
    tt::tt_metal::KernelHandle sdpa_fw_kernel_group_2{};
    tt::tt_metal::CoreRangeSet core_group_1{};
    tt::tt_metal::CoreRangeSet core_group_2{};
    uint32_t num_cores{};
    uint32_t num_cores_y{};
};

struct RingSDPAFwProgramFactory {
    using shared_variables_t = RingSDPAFwSharedVariables;
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

// Create MeshWorkload for forward SDPA
// mask_type determines which mask is used per device:
// - None: No masking (full attention)
// - Causal: Uses causal mask for step 0, full for earlier chunks, skips later chunks
tt::tt_metal::distributed::MeshWorkload create_ring_sdpa_fw_workload(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    ttnn::Tensor& output,
    ttnn::Tensor& intermediates,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type);

}  // namespace ttml::metal::ops::ring_sdpa_fw
