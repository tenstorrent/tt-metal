// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gated_rmsnorm_device_operation_types.hpp"
#include "gated_rmsnorm_program_common.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gated_rmsnorm::device {

struct GatedRmsNormFwProgramFactory {
    using shared_variables_t = GatedRmsNormSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const fw::Params& operation_attributes, const fw::Inputs& tensor_args, fw::tensor_return_value_t& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const fw::Params& operation_attributes,
        const fw::Inputs& tensor_args,
        fw::tensor_return_value_t& output);
};

}  // namespace ttml::metal::ops::gated_rmsnorm::device
