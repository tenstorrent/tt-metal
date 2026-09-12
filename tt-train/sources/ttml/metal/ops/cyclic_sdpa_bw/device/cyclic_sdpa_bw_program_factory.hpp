// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "cyclic_sdpa_bw_device_operation_types.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cyclic_sdpa_bw::device {

// How one invocation is laid out on the grid. Derived from the tensor shapes
// and the device's own grid rather than passed in, because the schedule
// leaves no freedom: C follows from the sequence length and the block height,
// and the group rectangle has to have area exactly C for every edge of the
// parity snake to be one hop.
struct CyclicLayout {
    uint32_t cores_per_group{};   // C
    uint32_t group_width{};       // grid_w, with group_width * group_height == C
    uint32_t group_height{};      // grid_h
    uint32_t groups{};            // one per (batch, head) slice
    uint32_t groups_across{};     // how many fit along x before wrapping
    std::vector<tt::tt_metal::CoreCoord> group_origin;
    tt::tt_metal::CoreRangeSet region;  // the union, never the bounding box
};

CyclicLayout plan_layout(
    const tt::tt_metal::CoreCoord& compute_grid,
    uint32_t sequence_length,
    uint32_t rows_per_block_tiles,
    uint32_t slices);

struct CyclicSDPABackwardProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        tt::tt_metal::KernelHandle compute_kernel_id{};
        CyclicLayout layout;
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

}  // namespace ttml::metal::ops::cyclic_sdpa_bw::device
