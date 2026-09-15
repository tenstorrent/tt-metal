// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ring_zigzag_sdpa: the ring_sdpa_fw / ring_sdpa_bw mesh wrapper over the
// single-chip SDPA kernels, with the chips that run chosen by where this
// step's visiting chunk comes from (ops::ZigzagVisitor) instead of by the
// contiguous layout's causal rule. The kernels and the reference ring ops
// are untouched; this exists so the zigzag layout can use them on one chunk
// pair per launch.

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ring_zigzag_bw_q_device_operation_types.hpp"

namespace ttml::metal::ops::ring_zigzag_bw::q {

// Backward Q Program Factory
struct RingZigzagBwQSharedVariables {
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

struct RingZigzagBwQProgramFactory {
    using shared_variables_t = RingZigzagBwQSharedVariables;
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
}  // namespace ttml::metal::ops::ring_zigzag_bw::q
