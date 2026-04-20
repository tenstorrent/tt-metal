// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "polynorm_bw_device_operation_types.hpp"

namespace ttml::metal::ops::polynorm3_bw::device {

// Program factory for PolyNorm3 backward (reader/compute/writer kernels + runtime args).
struct PolyNorm3BackwardProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle dL_dx_writer_kernel_id{};
        tt::tt_metal::KernelHandle compute_kernel_group_1_id{};
        tt::tt_metal::KernelHandle compute_kernel_group_2_id{};
        tt::tt_metal::CoreRangeSet core_group_1;
        tt::tt_metal::CoreRangeSet core_group_2;
        uint32_t num_cores{};
        uint32_t num_cores_y{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttml::metal::ops::polynorm3_bw::device
