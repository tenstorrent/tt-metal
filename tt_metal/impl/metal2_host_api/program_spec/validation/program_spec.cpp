// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * This file contains validation of ProgramSpec struct that's not covered witin
 * `../resource` or `../placement`.
 *
 * Normally those files contain the most validations, and current program_spec.cpp is just a catch-all for everything
 * else.
 */

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/program_spec/validation/validate_spec.hpp"

namespace tt::tt_metal::experimental {

void ValidateProgramMisc(const ValidationContext& ctx) {
    const ProgramSpec& spec = ctx.spec;
    const CollectedSpecData& collected = ctx.collected;

    // A Program needs at least one kernel
    TT_FATAL(!spec.kernels.empty(), "A ProgramSpec must have at least one KernelSpec");

    // Every declared cross-node DFB must be bound by some kernel (local DFBs are checked in collection).
    for (const auto& cross_node_dfb : spec.cross_node_dataflow_buffers) {
        const DFBSpecName& name = cross_node_dfb.dfb_spec.unique_id;
        TT_FATAL(
            collected.dfb_endpoints.contains(name),
            "CrossNodeDataflowBufferSpec '{}' is defined but not bound by any kernel",
            name);
    }
    for (const auto& cross_node_dfb : spec.cross_node_dataflow_buffers) {
        TT_FATAL(
            cross_node_dfb.dfb_spec.advanced_options.prefetcher_pipe_relays.empty(),
            "CrossNodeDataflowBufferSpec '{}' sets prefetcher_pipe_relays; only a local DFB can relay a "
            "PrefetcherPipe",
            cross_node_dfb.dfb_spec.unique_id);
    }

    // Cross-node DFBs are not yet supported.
    //
    // TODO: When cross-node DFB is supported, add a validation checks. Enforce that
    //       each (producer_node, consumer_node) entry in producer_consumer_map has
    //       p_node != c_node.

    TT_FATAL(
        spec.cross_node_dataflow_buffers.empty(),
        "CrossNodeDataflowBufferSpec is part of the Metal 2.0 API surface but is not yet supported "
        "by the runtime. (ProgramSpec '{}' has {} cross-node DFB(s).)",
        spec.name,
        spec.cross_node_dataflow_buffers.size());
}

}  // namespace tt::tt_metal::experimental
