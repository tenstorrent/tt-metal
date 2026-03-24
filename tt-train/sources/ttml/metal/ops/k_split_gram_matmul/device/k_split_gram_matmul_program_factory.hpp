// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/device_operation.hpp>

#include "k_split_gram_matmul_device_operation_types.hpp"

namespace ttml::metal::ops::k_split_gram_matmul::device {

struct KSplitGramMatmulProgramFactory {
    struct shared_variables_t {
        uint32_t dummy{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& output);

    static void override_runtime_arguments(
        cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

}  // namespace ttml::metal::ops::k_split_gram_matmul::device
