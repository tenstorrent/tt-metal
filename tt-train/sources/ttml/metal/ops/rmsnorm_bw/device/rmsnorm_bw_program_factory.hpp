// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "rmsnorm_bw_device_operation_types.hpp"

namespace ttml::metal::ops::rmsnorm_bw::device {

struct RMSNormBackwardSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<tt::tt_metal::CoreCoord> cores;
};

struct RMSNormBackwardPartialProgramFactory {
    using shared_variables_t = RMSNormBackwardSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const partial::operation_attributes_t& operation_attributes,
        const partial::tensor_args_t& tensor_args,
        partial::tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const partial::operation_attributes_t& operation_attributes,
        const partial::tensor_args_t& tensor_args,
        partial::tensor_return_value_t& tensor_return_value);
};

struct RMSNormBackwardProgramFactory {
    using shared_variables_t = RMSNormBackwardSharedVariables;
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
