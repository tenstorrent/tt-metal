// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace tt::tt_fabric {

class FabricContext;

// Private worker connection API on a chosen VC -- Metal-layer only, not published under public fabric API.
// Connects the worker to the VC's worker channel (its first sender channel) on the router for (src -> dst, link):
// VC2, or a 1D fabric's line VCs (TT_METAL_FABRIC_LINE_VCS). Only supports CoreType::WORKER.
template <typename ProgramOrDescriptor = tt::tt_metal::Program>
void append_fabric_connection_rt_args_on_vc(
    const FabricContext& fabric_context,
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    uint32_t link_idx,
    uint32_t vc,
    ProgramOrDescriptor& worker_program_or_desc,
    const tt::tt_metal::CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args);

// VC2's: append_fabric_connection_rt_args_on_vc with vc = 2.
template <typename ProgramOrDescriptor = tt::tt_metal::Program>
void append_fabric_vc2_connection_rt_args(
    const FabricContext& fabric_context,
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    uint32_t link_idx,
    ProgramOrDescriptor& worker_program_or_desc,
    const tt::tt_metal::CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args);

}  // namespace tt::tt_fabric
