// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gram_polynomial_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gram_polynomial::device {

struct HxPlusAxProgramFactory {
    struct shared_variables_t {
        uint32_t num_cores{};
        std::vector<tt::tt_metal::CoreCoord> cores;
        tt::tt_metal::KernelHandle in0_sender_kernels_id{};
        tt::tt_metal::KernelHandle in0_receiver_kernels_id{};
        tt::tt_metal::KernelHandle in1_sender_kernels_id{};
        tt::tt_metal::KernelHandle in1_receiver_kernels_id{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const phase3_operation_attributes_t& operation_attributes,
        const phase3_tensor_args_t& tensor_args,
        phase3_tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const phase3_operation_attributes_t& operation_attributes,
        const phase3_tensor_args_t& tensor_args,
        phase3_tensor_return_value_t& tensor_return_value);
};

}  // namespace ttml::metal::ops::gram_polynomial::device
