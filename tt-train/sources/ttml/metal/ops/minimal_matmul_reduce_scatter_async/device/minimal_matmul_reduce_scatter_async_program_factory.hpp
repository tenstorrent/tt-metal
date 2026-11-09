// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "minimal_matmul_reduce_scatter_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::minimal_matmul_reduce_scatter_async {
struct MinimalMatmulReduceScatterAsyncProgramFactory {
    struct shared_variables_t {
        struct CoreKernels {
            ttnn::CoreCoord core;
            std::optional<tt::tt_metal::KernelHandle> in0;
            std::optional<tt::tt_metal::KernelHandle> in1;
        };
        std::vector<CoreKernels> cores;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const MinimalMatmulReduceScatterAsyncParams& operation_attributes,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const MinimalMatmulReduceScatterAsyncInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const MinimalMatmulReduceScatterAsyncParams& operation_attributes,
        const MinimalMatmulReduceScatterAsyncInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};
}  // namespace ttml::metal::ops::minimal_matmul_reduce_scatter_async
