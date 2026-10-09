// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gated_rmsnorm_device_operation_types.hpp"
#include "gated_rmsnorm_program_common.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gated_rmsnorm::device {

struct GatedRmsNormBwProgramFactory {
    using shared_variables_t = GatedRmsNormSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const bw::Params& operation_attributes, const bw::Inputs& tensor_args, bw::tensor_return_value_t& outputs);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const bw::Params& operation_attributes,
        const bw::Inputs& tensor_args,
        bw::tensor_return_value_t& outputs);
};

}  // namespace ttml::metal::ops::gated_rmsnorm::device
