// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gated_rmsnorm_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::gated_rmsnorm::device {

struct GatedRmsNormBackwardProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        std::vector<tt::tt_metal::CoreCoord> cores;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const bw::operation_attributes_t& operation_attributes,
        const bw::tensor_args_t& tensor_args,
        bw::tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const bw::operation_attributes_t& operation_attributes,
        const bw::tensor_args_t& tensor_args,
        bw::tensor_return_value_t& tensor_return_value);
};

}  // namespace ttml::metal::ops::gated_rmsnorm::device
