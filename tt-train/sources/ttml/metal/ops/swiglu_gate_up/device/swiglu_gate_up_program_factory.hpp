// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "swiglu_gate_up_device_operation_types.hpp"

namespace ttml::metal::ops::swiglu_gate_up::device {

struct SwiGLUGateUpProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle in0_sender_kernel_id;
        tt::tt_metal::KernelHandle in0_receiver_kernel_id;
        tt::tt_metal::KernelHandle in1_sender_writer_kernel_id;
        tt::tt_metal::KernelHandle in1_receiver_writer_kernel_id;
        tt::tt_metal::KernelHandle compute_kernel_id;
        uint32_t num_cores_r{};
        uint32_t num_cores_c{};
        uint32_t per_core_M{};
        uint32_t per_core_N{};
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

}  // namespace ttml::metal::ops::swiglu_gate_up::device
