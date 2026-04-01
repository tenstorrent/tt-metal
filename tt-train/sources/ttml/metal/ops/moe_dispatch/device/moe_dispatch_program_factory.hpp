// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/types.hpp>

#include "moe_dispatch_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::moe_dispatch {

struct MoeDispatchMeshWorkloadFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle sender_kid;
        tt::tt_metal::KernelHandle recv_reader_kid;
        tt::tt_metal::KernelHandle compute_kid;
        tt::tt_metal::KernelHandle writer_kid;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const MoeDispatchParams& attrs,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const MoeDispatchTensorArgs& tensor_args,
        std::vector<ttnn::Tensor>& output);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const MoeDispatchParams& attrs,
        const ttnn::MeshCoordinate& coord,
        const MoeDispatchTensorArgs& tensor_args,
        std::vector<ttnn::Tensor>& output,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached,
        const MoeDispatchParams& attrs,
        const MoeDispatchTensorArgs& tensor_args,
        std::vector<ttnn::Tensor>& output);
};

}  // namespace ttml::metal::ops::moe_dispatch
