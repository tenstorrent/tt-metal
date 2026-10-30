// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "add_rmsnorm_fw_allcore_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::add_rmsnorm_fw_allcore::device {

struct AddRMSNormForwardSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<tt::tt_metal::CoreCoord> cores;
};

struct AddRMSNormForwardPartialProgramFactory {
    using shared_variables_t = AddRMSNormForwardSharedVariables;
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

struct AddRMSNormForwardProgramFactory {
    using shared_variables_t = AddRMSNormForwardSharedVariables;
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

}  // namespace ttml::metal::ops::add_rmsnorm_fw_allcore::device
