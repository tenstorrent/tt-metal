// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "chunk_gated_delta_rule_fw_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::chunk_gated_delta_rule_fw::device {

// One circular buffer of a program: index, tile count, and whether it holds BFLOAT16 input tiles
// (otherwise fp32).
struct CbSpec {
    uint32_t index{};
    uint32_t tiles{};
    bool bf16{};
};

// The circular buffers of each program in creation order, shared by the factories and the L1 check.
std::vector<CbSpec> prep_cb_plan(uint32_t Ct, uint32_t Kt, uint32_t Vt);
std::vector<CbSpec> scan_cb_plan(uint32_t Ct, uint32_t Kt, uint32_t Vt_block);
uint32_t cb_plan_bytes(const std::vector<CbSpec>& plan);
// L1 available to circular buffers on a worker core.
uint32_t cb_l1_budget();

// V-parallel split of the scan: each of BH * num_vblocks cores runs one (value head, V-block) scan.
// The largest num_vblocks dividing Vt that fits the grid.
struct ScanSplit {
    uint32_t num_vblocks{};
    uint32_t vblock_tiles{};
};
ScanSplit make_scan_split(uint32_t BH, uint32_t Vt, uint32_t num_cores);

struct ChunkGatedDeltaRuleForwardPrepProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        std::vector<tt::tt_metal::CoreCoord> cores;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ChunkGatedDeltaRuleForwardPrepParams& operation_attributes,
        const ChunkGatedDeltaRuleForwardPrepInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ChunkGatedDeltaRuleForwardPrepParams& operation_attributes,
        const ChunkGatedDeltaRuleForwardPrepInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

struct ChunkGatedDeltaRuleForwardScanProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        std::vector<tt::tt_metal::CoreCoord> cores;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ChunkGatedDeltaRuleForwardScanParams& operation_attributes,
        const ChunkGatedDeltaRuleForwardScanInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ChunkGatedDeltaRuleForwardScanParams& operation_attributes,
        const ChunkGatedDeltaRuleForwardScanInputs& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttml::metal::ops::chunk_gated_delta_rule_fw::device
