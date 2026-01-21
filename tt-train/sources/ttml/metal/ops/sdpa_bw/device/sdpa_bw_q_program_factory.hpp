// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "sdpa_bw_q_device_operation_types.hpp"

namespace ttml::metal::ops::sdpa_bw::device {

struct SDPABackwardQProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle sdpa_bw_q_reader_kernel{};
        tt::tt_metal::KernelHandle sdpa_bw_q_writer_kernel{};
        tt::tt_metal::KernelHandle sdpa_bw_q_kernel_group_1{};
        tt::tt_metal::KernelHandle sdpa_bw_q_kernel_group_2{};
        tt::tt_metal::CoreRangeSet core_group_1;
        tt::tt_metal::CoreRangeSet core_group_2;
        uint32_t num_cores{};
        uint32_t num_cores_y{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const q::operation_attributes_t& operation_attributes,
        const q::tensor_args_t& tensor_args,
        q::tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const q::operation_attributes_t& operation_attributes,
        const q::tensor_args_t& tensor_args,
        q::tensor_return_value_t& tensor_return_value);
};

}  // namespace ttml::metal::ops::sdpa_bw::device
