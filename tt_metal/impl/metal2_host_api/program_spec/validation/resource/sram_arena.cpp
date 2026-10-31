// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_set>
#include <variant>

#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"

namespace tt::tt_metal::experimental {

void ValidateSRAMArenaParameter(const SRAMArenaParameter& parameter) {
    const auto per_bank_size =
        std::visit([](const auto& properties) { return properties.per_bank_size; }, parameter.properties);
    TT_FATAL(
        per_bank_size > 0,
        "SRAMArenaParameter '{}' has per_bank_size = 0; per_bank_size must be greater than zero.",
        parameter.unique_id);
    if (const auto* range = std::get_if<RangeLockstepSRAMArenaProperties>(&parameter.properties)) {
        TT_FATAL(
            !range->banks_to_be_allocated_on.empty(),
            "SRAMArenaParameter '{}' has empty banks_to_be_allocated_on; at least one node is required.",
            parameter.unique_id);
    }
}

void ValidateSRAMArenaAllocation(
    const SRAMArenaSource& source,
    const CollectedSpecData& collected,
    std::string_view resource,
    const NodeRangeSet& occupied_nodes,
    uint64_t size_per_node,
    uint32_t l1_alignment) {
    const auto it = collected.sram_arena_parameter_by_name.find(source.arena);
    TT_FATAL(
        it != collected.sram_arena_parameter_by_name.end(),
        "{} allocates from unknown SRAMArenaParameter '{}'.",
        resource,
        source.arena);
    TT_FATAL(
        source.offset % l1_alignment == 0,
        "{} allocation offset {} must be a multiple of L1 alignment {}.",
        resource,
        source.offset,
        l1_alignment);

    const auto& properties = it->second->properties;
    if (const auto* range = std::get_if<RangeLockstepSRAMArenaProperties>(&properties)) {
        TT_FATAL(
            range->banks_to_be_allocated_on.contains(occupied_nodes),
            "{} allocates from SRAMArenaParameter '{}', which does not cover every occupied node.",
            resource,
            source.arena);
    }
    ValidateSRAMArenaCapacity(source, properties, resource, size_per_node);
}

void ValidateSRAMArenaParametersUsed(const ValidationContext& ctx) {
    std::unordered_set<SRAMArenaParamName> used_arenas;
    for (const auto& dfb : ctx.spec.dataflow_buffers) {
        if (const auto& allocation = dfb.advanced_options.allocation_override) {
            used_arenas.insert(allocation->arena);
        }
    }
    for (const auto& scratchpad : ctx.spec.scratchpads) {
        if (const auto& allocation = scratchpad.advanced_options.allocation_override) {
            used_arenas.insert(allocation->arena);
        }
    }
    for (const auto& parameter : ctx.spec.advanced_options.sram_arena_parameters) {
        TT_FATAL(
            used_arenas.contains(parameter.unique_id),
            "SRAMArenaParameter '{}' is not used by any DFB or scratchpad.",
            parameter.unique_id);
    }
}

}  // namespace tt::tt_metal::experimental
