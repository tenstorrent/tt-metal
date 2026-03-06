// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ring_sdpa_bw_q_device_operation_types.hpp"

namespace ttml::metal::ops::ring_sdpa_bw::q {

// Backward Q Program Factory
struct RingSDPABwQSharedVariables {
    // SDPA backward Q kernel handles
    tt::tt_metal::KernelHandle sdpa_bw_q_reader_kernel{};
    tt::tt_metal::KernelHandle sdpa_bw_q_writer_kernel{};
    tt::tt_metal::KernelHandle sdpa_bw_q_kernel_group_1{};
    tt::tt_metal::KernelHandle sdpa_bw_q_kernel_group_2{};
    tt::tt_metal::CoreRangeSet core_group_1{};
    tt::tt_metal::CoreRangeSet core_group_2{};
    uint32_t num_cores{};
    uint32_t num_cores_y{};
};

struct RingSDPABwQProgramFactory {
    using shared_variables_t = RingSDPABwQSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        ttnn::Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        ttnn::Tensor& tensor_return_value);
};
}  // namespace ttml::metal::ops::ring_sdpa_bw::q
