// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "variable_matmul_device_operation_types.hpp"

namespace ttml::metal::ops::variable_matmul::device {

struct VariableMatmulProgramFactory {
    struct shared_variables_t {
        uint32_t num_cores{};
        std::vector<tt::tt_metal::CoreCoord> cores;
        tt::tt_metal::KernelHandle in0_sender_kernels_id{};
        tt::tt_metal::KernelHandle in0_receiver_kernels_id{};
        tt::tt_metal::KernelHandle in1_sender_kernels_id{};
        tt::tt_metal::KernelHandle in1_receiver_kernels_id{};
        tt::tt_metal::KernelHandle compute_kernels_id{};
        bool transpose_core_grid{};
        // For recomputing M-dependent runtime args on cache hit
        uint32_t in0_parallel_axis_cores{};
        uint32_t in1_parallel_axis_cores{};
        uint32_t M_block_tiles{};
        uint32_t K_block_tiles{};
        uint32_t N_tiles_per_core{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const VariableMatmulParams& operation_attributes,
        const VariableMatmulInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const VariableMatmulParams& operation_attributes,
        const VariableMatmulInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttml::metal::ops::variable_matmul::device
