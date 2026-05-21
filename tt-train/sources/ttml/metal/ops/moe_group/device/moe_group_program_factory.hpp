// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "moe_group_device_operation_types.hpp"

namespace ttml::metal::ops::moe_group::device {

struct MoeGroupProgramFactory {
    struct shared_variables_t {
        uint32_t num_cores{};
        tt::tt_metal::CoreCoord scan_coord{};
        tt::tt_metal::KernelHandle scan_kernel{};
        tt::tt_metal::KernelHandle reader_kernel_g1{};
        tt::tt_metal::KernelHandle reader_kernel_g2{};
        tt::tt_metal::KernelHandle writer_kernel_g1{};
        tt::tt_metal::KernelHandle writer_kernel_g2{};
        tt::tt_metal::CoreRangeSet worker_all;
        tt::tt_metal::CoreRangeSet worker_group_1;
        tt::tt_metal::CoreRangeSet worker_group_2;
        uint32_t num_cores_y{};
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

}  // namespace ttml::metal::ops::moe_group::device
