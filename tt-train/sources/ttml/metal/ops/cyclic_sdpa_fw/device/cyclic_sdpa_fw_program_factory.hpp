// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cyclic_sdpa_fw_device_operation_types.hpp"
#include "metal/ops/cyclic_sdpa_bw/device/cyclic_sdpa_bw_program_factory.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::cyclic_sdpa_fw::device {

// The grid layout is the backward's: C follows from the chunk length and the
// block height, groups are rectangles of area C embedding the parity snake,
// slices are dealt round-robin to the groups.
using ttml::metal::ops::cyclic_sdpa_bw::device::CyclicLayout;
using ttml::metal::ops::cyclic_sdpa_bw::device::plan_layout;
using ttml::metal::ops::cyclic_sdpa_bw::device::slices_of_group;

struct CyclicSDPAForwardProgramFactory {
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

}  // namespace ttml::metal::ops::cyclic_sdpa_fw::device
