// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cstdint>
#include <vector>

#include "gumbel_sample_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gumbel_sample::device {

// `rand_tile` is documented as inclusive of `from + scale`. Shrink the scale by one ULP if rounding
// would push the top of the range past the intended upper bound.
inline uint32_t compute_rand_scale_bits(float lower, float upper) {
    float scale = upper - lower;
    uint32_t scale_bits = std::bit_cast<uint32_t>(scale);
    if (lower + scale > upper && scale_bits != 0U) {
        --scale_bits;
    }
    return scale_bits;
}

struct GumbelSampleSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_group_1_id{};
    tt::tt_metal::KernelHandle compute_kernel_group_2_id{};
    bool has_compute_group_2{};
    // Everything the cache-hit patch needs per core, derived once at build: the work split is a
    // function of hashed quantities only, and this op dispatches once per generated token, so
    // re-deriving it per hit would be paid thousands of times per rollout. Caching also
    // single-sources the stream-id derivation, whose divergence would only show on cache hits.
    struct CoreRuntimeInfo {
        tt::tt_metal::CoreCoord core;
        uint32_t rand_stream_id{};
        bool in_compute_group_1{};
    };
    std::vector<CoreRuntimeInfo> core_info;
};

// Builds a MESH WORKLOAD (one program per mesh coordinate) rather than a single broadcast program:
// the RNG seed must differ per device on data-parallel axes. Same pattern as
// RingSDPAFwProgramFactory.
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
