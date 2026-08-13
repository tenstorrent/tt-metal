// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gumbel_sample_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gumbel_sample::device {

struct GumbelSampleSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_group_1_id{};
    tt::tt_metal::KernelHandle compute_kernel_group_2_id{};
    tt::tt_metal::CoreRangeSet core_group_1{};
    tt::tt_metal::CoreRangeSet core_group_2{};
    uint32_t num_cores{};
    uint32_t num_cores_y{};
    // Baked per-device RNG seed base. Kept so override_runtime_arguments can re-apply the seed on a
    // program-cache hit without re-deriving the mesh coordinate.
    uint32_t device_seed_offset{};
};

// NOTE: this factory builds a MESH WORKLOAD (one program per mesh coordinate) rather than a single
// program, because the RNG seed has to differ per device on data-parallel axes. The plain
// `create()` factories used by the other tt-train ops emit one program broadcast to every device,
// which would make every data-parallel replica draw identical noise and emit identical samples --
// exactly the GRPO duplicate-completion bug that ttnn_fixed::sample's seed_axes plumbing exists to
// avoid. See RingSDPAFwProgramFactory for the same pattern.
struct GumbelSampleProgramFactory {
    using shared_variables_t = GumbelSampleSharedVariables;
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

}  // namespace ttml::metal::ops::gumbel_sample::device
