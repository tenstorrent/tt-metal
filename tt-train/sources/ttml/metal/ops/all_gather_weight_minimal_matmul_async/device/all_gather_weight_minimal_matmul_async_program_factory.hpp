// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "all_gather_weight_minimal_matmul_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async {
struct AllGatherWeightMinimalMatmulAsyncProgramFactory {
    struct shared_variables_t {
        std::vector<ttnn::CoreCoord> cores;
        tt::tt_metal::KernelHandle in0_sender_kernels_id{};
        tt::tt_metal::KernelHandle in0_receiver_kernels_id{};
        tt::tt_metal::KernelHandle in1_sender_kernels_id{};
        tt::tt_metal::KernelHandle in1_receiver_kernels_id{};
        bool transpose_core_grid{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const AllGatherWeightMinimalMatmulAsyncParams& operation_attributes,
        const AllGatherWeightMinimalMatmulAsyncInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const AllGatherWeightMinimalMatmulAsyncParams& operation_attributes,
        const AllGatherWeightMinimalMatmulAsyncInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};
}  // namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async
