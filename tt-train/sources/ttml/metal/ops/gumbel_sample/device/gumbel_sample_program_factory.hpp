// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "gumbel_sample_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gumbel_sample::device {

struct GumbelSampleSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_group_1_id{};
    tt::tt_metal::KernelHandle compute_kernel_group_2_id{};
    bool has_compute_group_2{};
    // Everything the cache-hit patch needs per core, derived ONCE at build time. All of it is
    // invariant across hits of one cached program: the work split is a function of hashed
    // quantities only (padded dims, device grid), and the RNG stream id folds the device index
    // and the core's start tile -- both split properties. This op dispatches once per generated
    // token, so re-deriving the split on every hit (device grid query, split_work_to_cores,
    // per-core CoreRangeSet scans) would be paid thousands of times per rollout on the host
    // dispatch path; caching also single-sources the stream-id derivation, whose divergence
    // between build and patch would manifest only on cache hits -- which single-shape unit tests
    // never exercise.
    struct CoreRuntimeInfo {
        tt::tt_metal::CoreCoord core;
        uint32_t rand_stream_id{};
        bool in_compute_group_1{};
    };
    std::vector<CoreRuntimeInfo> core_info;
};

// NOTE: this factory builds a MESH WORKLOAD (one program per mesh coordinate) rather than a single
// program, because the RNG seed has to differ per device on data-parallel axes. The plain
// `create()` factories used by the other tt-train ops emit one program broadcast to every device,
// which would make every data-parallel replica draw identical noise and emit identical samples.
// See RingSDPAFwProgramFactory for the same pattern.
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
