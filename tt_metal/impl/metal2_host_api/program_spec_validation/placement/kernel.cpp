// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include <utility>

#include <tt_stl/assert.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec_validation/placement/placement.hpp"

namespace tt::tt_metal::experimental {

// On Gen1 (WH/BH), the DM kernels sharing a node must be mutually coherent:
//   1. Distinct DM processors (RISCV_0 vs RISCV_1) — no two kernels may pin the same RISC.
//   2. Agreeing NOC mode. noc_mode configures shared per-core NOC hardware (command-buffer
//      partitioning + completion-counter location) and is compiled into each kernel binary as
//      the NOC_MODE define (see kernel.cpp), so two kernels on a node with different modes are
//      incoherent. Mirrors the KernelGroup-construction guard ("KernelGroup must have the same
//      noc mode for all kernels"), surfaced here at spec-validation time with a clearer message.
//   3. In DM_DEDICATED_NOC mode, distinct NOCs. Each DM kernel's NoC traffic is statically
//      compiled to NOC_INDEX == config.noc (see kernel.cpp), so two dedicated-NOC kernels
//      pinned to the same NOC deadlock the device. This enforces the NOC-distinctness invariant
//      that KernelGroup finalize silently relies on (it writes brisc_noc_id = arg.noc for
//      RISCV_0 vs 1 - arg.noc for RISCV_1, which agree only when the two NOCs differ -- "safe
//      due to prior correctness validation"). The legacy CheckDataMovementConfig intended this
//      check but did not reliably enforce it for the common reader+writer pair (it runs before
//      the second DM kernel is registered). DM_DYNAMIC_NOC kernels are exempt: they may
//      intentionally share a NOC, freeing the other NOC for fabric.
// (Each kernel's effective node set is derived from WorkUnitSpec membership.)
void ValidateGen1DMPlacement(const ValidationContext& ctx, tt::ARCH arch) {
    const ProgramSpec& spec = ctx.spec;
    const CollectedSpecData& collected = ctx.collected;

    if (!is_gen1_arch(arch)) {
        return;
    }
    // (node, processor) -> the kernel that already claimed it.
    std::map<std::pair<NodeCoord, DataMovementProcessor>, KernelSpecName> claimed_processor;
    // node -> (noc mode, the kernel that first set it) — all DM kernels on a node must agree.
    std::map<NodeCoord, std::pair<NOC_MODE, KernelSpecName>> node_noc_mode;
    // (node, noc) -> the kernel that already claimed it (dedicated-NOC kernels only).
    std::map<std::pair<NodeCoord, NOC>, KernelSpecName> claimed_noc;
    for (const auto& kernel : spec.kernels) {
        if (!kernel.is_data_movement_kernel()) {
            continue;
        }
        const auto& dm_config = std::get<DataMovementHardwareConfig>(kernel.hw_config);
        if (!dm_config.config_1xx.has_value()) {
            continue;
        }
        const auto& gen1 = *dm_config.config_1xx;
        const NodeRangeSet& nodes = collected.kernel_node_set.at(kernel.unique_id);
        for (const auto& range : nodes.ranges()) {
            for (const auto& node : range) {
                auto [proc_it, proc_inserted] =
                    claimed_processor.try_emplace(std::make_pair(node, gen1.processor), kernel.unique_id);
                TT_FATAL(
                    proc_inserted,
                    "KernelSpec '{}' conflicts with '{}' on node ({}, {}): both claim the same DM processor. ",
                    kernel.unique_id,
                    proc_it->second,
                    node.x,
                    node.y);

                // All DM kernels on a node must agree on NOC mode -- it configures shared per-core NOC
                // hardware. Independent of the NOC-distinctness check below (which is gated per-kernel on
                // DM_DEDICATED_NOC); their source order does not affect behavior.
                auto [mode_it, mode_inserted] =
                    node_noc_mode.try_emplace(node, std::make_pair(gen1.noc_mode, kernel.unique_id));
                TT_FATAL(
                    mode_inserted || mode_it->second.first == gen1.noc_mode,
                    "KernelSpec '{}' conflicts with '{}' on node ({}, {}): they set different NOC modes (one "
                    "DM_DEDICATED_NOC, the other DM_DYNAMIC_NOC). All data movement kernels on a node must use "
                    "the same NOC mode.",
                    kernel.unique_id,
                    mode_it->second.second,
                    node.x,
                    node.y);

                // NOC-distinctness applies only to statically-pinned (dedicated-NOC) kernels.
                if (gen1.noc_mode == NOC_MODE::DM_DEDICATED_NOC) {
                    auto [noc_it, noc_inserted] =
                        claimed_noc.try_emplace(std::make_pair(node, gen1.noc), kernel.unique_id);
                    TT_FATAL(
                        noc_inserted,
                        "KernelSpec '{}' conflicts with '{}' on node ({}, {}): both are dedicated-NOC data "
                        "movement kernels pinned to NOC_{}, which hangs the device. Give them distinct NOCs, or "
                        "use DM_DYNAMIC_NOC mode to intentionally share a NOC.",
                        kernel.unique_id,
                        noc_it->second,
                        node.x,
                        node.y,
                        static_cast<int>(gen1.noc));
                }
            }
        }
    }
}

}  // namespace tt::tt_metal::experimental
