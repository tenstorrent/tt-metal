// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ring_sdpa_bw_kv_device_operation_types.hpp"

namespace ttml::metal::ops::ring_sdpa_bw::kv {

// Backward KV Program Factory
struct RingSDPABwKVSharedVariables {
    // SDPA backward KV kernel handles
    tt::tt_metal::KernelHandle sdpa_bw_reader_kernel{};
    tt::tt_metal::KernelHandle sdpa_bw_writer_kernel{};
    tt::tt_metal::KernelHandle sdpa_bw_kernel_group_1{};
    tt::tt_metal::KernelHandle sdpa_bw_kernel_group_2{};
    tt::tt_metal::CoreRangeSet core_group_1{};
    tt::tt_metal::CoreRangeSet core_group_2{};
    uint32_t num_cores{};
    uint32_t num_cores_y{};
};

struct RingSDPABwKVProgramFactory {
    using shared_variables_t = RingSDPABwKVSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value);
};

// Create MeshWorkload for backward KV gradient computation
// mask_type determines which mask is used per device:
// - None: No masking (full attention)
// - Causal: Uses causal mask for step 0, full for earlier chunks, skips later chunks
tt::tt_metal::distributed::MeshWorkload create_ring_sdpa_bw_kv_workload(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& intermediates,
    ttnn::Tensor& grad_key,
    ttnn::Tensor& grad_value,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type);

}  // namespace ttml::metal::ops::ring_sdpa_bw::kv
