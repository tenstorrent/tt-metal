// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "skinny_matmul_device_operation_types.hpp"

namespace ttml::metal::ops::skinny_matmul::device {

struct SkinnyMatmulSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<tt::tt_metal::CoreCoord> cores;
};

struct SkinnyMatmulPartialProgramFactory {
    using shared_variables_t = SkinnyMatmulSharedVariables;
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

struct SkinnyMatmulReduceProgramFactory {
    using shared_variables_t = SkinnyMatmulSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const reduce::operation_attributes_t& operation_attributes,
        const reduce::tensor_args_t& tensor_args,
        reduce::tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const reduce::operation_attributes_t& operation_attributes,
        const reduce::tensor_args_t& tensor_args,
        reduce::tensor_return_value_t& tensor_return_value);
};

}  // namespace ttml::metal::ops::skinny_matmul::device
