// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "skinny_contract_device_operation_types.hpp"

namespace ttml::metal::ops::skinny_contract::device {

struct SkinnyContractProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};  // the multicast sender core's reader
        tt::tt_metal::KernelHandle reader_receiver_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        std::vector<tt::tt_metal::CoreCoord> cores;
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

}  // namespace ttml::metal::ops::skinny_contract::device
