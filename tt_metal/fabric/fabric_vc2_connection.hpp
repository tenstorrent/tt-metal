// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace tt::tt_fabric {

// Private VC2 connection API -- Metal-layer only, not published under public fabric API.
// Uses VC2 sender channel (last flat index) instead of channel 0.
// Only supports CoreType::WORKER (VC2 is for worker injection only).
template <typename ProgramOrDescriptor = tt::tt_metal::Program>
void append_fabric_vc2_connection_rt_args(
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    uint32_t link_idx,
    ProgramOrDescriptor& worker_program_or_desc,
    const CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args);

}  // namespace tt::tt_fabric
