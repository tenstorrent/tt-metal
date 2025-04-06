// SPDX-FileCopyrightText: © 205 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt-metalium/core_coord.hpp"
#include "dev_msgs.h"
#include "tt-metalium/kernel_types.hpp"      // KernelHandle
#include <umd/device/tt_core_coordinates.h>  // CoreType

#include <array>
#include <cstdint>
#include <optional>

namespace tt {
namespace tt_metal {

namespace detail {
class Program_;
}

typedef std::array<std::optional<KernelHandle>, DISPATCH_CLASS_MAX> kernel_id_array_t;

// Not used in ttnn
// Used in other public headers (mesh_workload.hpp)
struct KernelGroup {
    uint32_t programmable_core_type_index;
    CoreRangeSet core_ranges;
    kernel_id_array_t kernel_ids;
    uint32_t rta_sizes[DISPATCH_CLASS_MAX];
    uint32_t total_rta_size;
    uint32_t kernel_text_offsets[NUM_PROCESSORS_PER_CORE_TYPE];
    uint32_t kernel_bin_sizes[NUM_PROCESSORS_PER_CORE_TYPE];
    launch_msg_t launch_msg;
    go_msg_t go_msg;

    KernelGroup();
    KernelGroup(
        const detail::Program_& program,
        uint32_t programmable_core_type_index,
        kernel_id_array_t kernel_ids,
        bool erisc_is_idle,
        uint32_t max_local_cb_end_index,
        uint32_t min_remote_cb_start_index,
        const CoreRangeSet& new_ranges);

    uint32_t get_programmable_core_type_index() const;

    CoreType get_core_type() const;
};

}  // namespace tt_metal
}  // namespace tt
