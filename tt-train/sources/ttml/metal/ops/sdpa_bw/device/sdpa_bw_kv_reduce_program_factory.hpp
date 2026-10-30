// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "sdpa_bw_kv_device_operation_types.hpp"

namespace ttml::metal::ops::sdpa_bw::device {

struct SDPABackwardKVReduceProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel{};
        tt::tt_metal::KernelHandle writer_kernel{};
        uint32_t num_cores{};
        uint32_t num_cores_y{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const kv_reduce::operation_attributes_t& operation_attributes,
        const kv_reduce::tensor_args_t& tensor_args,
        kv_reduce::tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const kv_reduce::operation_attributes_t& operation_attributes,
        const kv_reduce::tensor_args_t& tensor_args,
        kv_reduce::tensor_return_value_t& tensor_return_value);
};

}  // namespace ttml::metal::ops::sdpa_bw::device
