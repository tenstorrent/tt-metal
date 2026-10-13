// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace tt::tt_fabric {

class FabricContext;

// Context-explicit form of the public append_fabric_connection_rt_args. Metal-layer only, not published under the
// public fabric API.
//
// The public form resolves the fabric context from the default MetalContext. This one reads everything from
// `fabric_context` (its control plane and cluster), so a caller that already holds its own context, such as the
// dispatch relay mux, stays correct when more than one context is live.
template <typename ProgramOrDescriptor = tt::tt_metal::Program>
void append_fabric_connection_rt_args_in_context(
    const FabricContext& fabric_context,
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    uint32_t link_idx,
    ProgramOrDescriptor& worker_program_or_desc,
    const tt::tt_metal::CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args,
    CoreType core_type = CoreType::WORKER);

}  // namespace tt::tt_fabric
