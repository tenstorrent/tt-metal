// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::metal::ops::ring_sdpa {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

// ============== Forward Pass Types ==============

struct RingSDPAParams {
    uint32_t ring_size;
    uint32_t ring_axis;
    uint32_t step;
    ttml::metal::AttentionMaskType mask_type;
    RingDirection ring_direction;  // Direction K/V is shifting in the ring
};

struct RingSDPAInputs {
    const ttnn::Tensor& query;
    const ttnn::Tensor& key;
    const ttnn::Tensor& value;
    std::optional<ttnn::Tensor> preallocated_output;         // Preallocated output buffer
    std::optional<ttnn::Tensor> preallocated_intermediates;  // Preallocated intermediates buffer
};

// Forward Program Factory
struct RingSDPASharedVariables {
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

struct RingSDPAProgramFactory {
    using shared_variables_t = RingSDPASharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const RingSDPAParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const RingSDPAInputs& tensor_args,
        std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const RingSDPAParams& operation_attributes,
        const RingSDPAInputs& tensor_args,
        std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value);
};

// ============== Backward Q Types ==============

struct RingSDPABwQParams {
    uint32_t ring_size;
    uint32_t ring_axis;
    uint32_t step;
    ttml::metal::AttentionMaskType mask_type;
    RingDirection ring_direction;  // Direction K/V is shifting in the ring
};

struct RingSDPABwQInputs {
    const ttnn::Tensor& grad_output;
    const ttnn::Tensor& attn_output;
    const ttnn::Tensor& query;
    const ttnn::Tensor& key;
    const ttnn::Tensor& value;
    const ttnn::Tensor& intermediates;
    std::optional<ttnn::Tensor> preallocated_grad_query;  // Preallocated output buffer
};

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
        const RingSDPABwQParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const RingSDPABwQInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const RingSDPABwQParams& operation_attributes,
        const RingSDPABwQInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

// ============== Backward KV Types ==============

struct RingSDPABwKVParams {
    uint32_t ring_size;
    uint32_t ring_axis;
    uint32_t step;
    ttml::metal::AttentionMaskType mask_type;
    RingDirection ring_direction;  // Direction K/V is shifting in the ring
};

struct RingSDPABwKVInputs {
    const ttnn::Tensor& grad_output;
    const ttnn::Tensor& attn_output;
    const ttnn::Tensor& query;
    const ttnn::Tensor& key;
    const ttnn::Tensor& value;
    const ttnn::Tensor& intermediates;
    std::optional<ttnn::Tensor> preallocated_grad_key;    // Preallocated output buffer
    std::optional<ttnn::Tensor> preallocated_grad_value;  // Preallocated output buffer
};

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
        const RingSDPABwKVParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const RingSDPABwKVInputs& tensor_args,
        std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const RingSDPABwKVParams& operation_attributes,
        const RingSDPABwKVInputs& tensor_args,
        std::tuple<ttnn::Tensor, ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttml::metal::ops::ring_sdpa
