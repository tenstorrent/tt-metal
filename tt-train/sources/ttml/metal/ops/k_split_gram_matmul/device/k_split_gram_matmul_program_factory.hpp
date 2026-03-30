// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/device_operation.hpp>

#include "k_split_gram_matmul_device_operation_types.hpp"

namespace ttml::metal::ops::k_split_gram_matmul::device {

struct KSplitGramMatmulProgramFactory {
    struct shared_variables_t {
        // Kernels with in_addr at runtime arg index 0
        tt::tt_metal::KernelHandle row_sender_reduce_kid;
        tt::tt_metal::KernelHandle row_sender_kid;
        tt::tt_metal::KernelHandle col_sender_kid;
        tt::tt_metal::KernelHandle helper_dram_reader_kid;
        // Kernel with out_addr at runtime arg index 2
        tt::tt_metal::KernelHandle row_upper_recv_kid;
        // Core lists for iterating runtime args
        std::vector<tt::tt_metal::CoreCoord> row_sender_cores;
        std::vector<tt::tt_metal::CoreCoord> col_sender_cores;
        std::vector<tt::tt_metal::CoreCoord> row_upper_cores;
        std::vector<tt::tt_metal::CoreCoord> helper_cores;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& output);

    static void override_runtime_arguments(
        cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

}  // namespace ttml::metal::ops::k_split_gram_matmul::device
