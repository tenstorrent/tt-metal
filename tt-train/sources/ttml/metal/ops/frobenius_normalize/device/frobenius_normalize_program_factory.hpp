// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "frobenius_normalize_device_operation_types.hpp"

namespace ttml::metal::ops::frobenius_normalize::device {

struct FrobeniusNormalizeProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle reader_origin_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        tt::tt_metal::KernelHandle compute_origin_id{};
        tt::tt_metal::KernelHandle compute_group_1_id{};
        tt::tt_metal::KernelHandle compute_group_2_id{};
        tt::tt_metal::CoreRangeSet core_group_1;
        tt::tt_metal::CoreRangeSet core_group_2;
        uint32_t num_cores{};
        uint32_t num_cores_y{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const FrobeniusNormalizeAttributes& operation_attributes,
        const FrobeniusNormalizeTensorArgs& tensor_args,
        FrobeniusNormalizeTensorReturn& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const FrobeniusNormalizeAttributes& operation_attributes,
        const FrobeniusNormalizeTensorArgs& tensor_args,
        FrobeniusNormalizeTensorReturn& tensor_return_value);
};

}  // namespace ttml::metal::ops::frobenius_normalize::device
