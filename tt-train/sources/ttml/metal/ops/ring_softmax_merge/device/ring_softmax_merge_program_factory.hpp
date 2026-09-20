// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"
#include "ring_softmax_merge_device_operation_types.hpp"

namespace ttml::metal::ops::ring_softmax_merge {

struct RingSoftmaxMergeSharedVariables {
    tt::tt_metal::KernelHandle reader{};
    tt::tt_metal::KernelHandle writer{};
    tt::tt_metal::KernelHandle compute{};
    std::vector<tt::tt_metal::CoreCoord> cores;  // the cores with work, in runtime-arg order
};

struct RingSoftmaxMergeProgramFactory {
    using shared_variables_t = RingSoftmaxMergeSharedVariables;
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

}  // namespace ttml::metal::ops::ring_softmax_merge
