// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "rmsnorm_bw_device_operation_types.hpp"

namespace ttml::metal::ops::rmsnorm_bw::device {

// Work split and L1 strategy. Shared by output-spec computation and program creation so that the dgamma output
// shape always matches what the kernels write.
struct RMSNormBackwardLayout {
    uint32_t num_cores{};
    bool everything_fits_in_l1{};
    // dgamma is accumulated per core in L1 and written as [1,1,num_cores*32,C] fp32 partials instead of
    // [B,N,S,C] per-row components.
    bool dgamma_on_chip{};
};

RMSNormBackwardLayout compute_layout(const operation_attributes_t& args, const tensor_args_t& tensor_args);

struct RMSNormBackwardProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle rmsnorm_bw_reader_kernel_id;
        tt::tt_metal::KernelHandle rmsnorm_bw_writer_kernel_id;
        tt::tt_metal::KernelHandle rmsnorm_bw_kernel_group_1_id;
        tt::tt_metal::KernelHandle rmsnorm_bw_kernel_group_2_id;
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

}  // namespace ttml::metal::ops::rmsnorm_bw::device
