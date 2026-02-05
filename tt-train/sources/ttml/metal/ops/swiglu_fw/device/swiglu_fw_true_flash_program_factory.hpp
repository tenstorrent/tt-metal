// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "swiglu_fw_device_operation_types.hpp"

namespace ttml::metal::ops::swiglu_fw::device {

// ============================================================================
// TRUE FLASH SwiGLU PROGRAM FACTORY
//
// This factory creates programs using the "True Flash" optimization that
// avoids materializing the full M row in L1. Instead, M tiles are computed
// on-demand for each k_block.
//
// Key differences from original:
//   - Loop order: k_block OUTER, p_block INNER (inverted from original)
//   - XW1_partial, XW3_partial, M: only block_size tiles (not full hidden_Wt)
//   - Y_partial, Y: full Wt tiles (accumulate all output columns)
//
// Memory savings: ~50% reduction for NanoLlama3 (560 KB → 280 KB)
// ============================================================================
struct SwiGLUTrueFlashProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_sender_kernel_id;
        tt::tt_metal::KernelHandle reader_receiver_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        tt::tt_metal::KernelHandle compute_kernel_group_1_id;
        tt::tt_metal::KernelHandle compute_kernel_group_2_id;
        tt::tt_metal::CoreRangeSet core_group_1;
        tt::tt_metal::CoreRangeSet core_group_2;
        uint32_t num_cores{};
        uint32_t num_cores_x{};
        uint32_t num_cores_y{};
        bool use_multicast{};
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

}  // namespace ttml::metal::ops::swiglu_fw::device
