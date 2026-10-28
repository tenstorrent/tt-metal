// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <unordered_set>

#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"

namespace tt::tt_metal::experimental {

void ValidateScratchpadSpec(
    const ScratchpadSpec& scratchpad, const CollectedSpecData& collected, uint32_t l1_alignment, tt::ARCH arch) {
    TT_FATAL(
        scratchpad.size_per_node != 0,
        "ScratchpadSpec '{}' has size_per_node == 0; a scratchpad must reserve a non-zero number of bytes "
        "(did you forget to set size_per_node?).",
        scratchpad.unique_id);

    const bool has_format = scratchpad.data_format_metadata.has_value();
    const bool has_tile = scratchpad.tile_format_metadata.has_value();
    TT_FATAL(
        has_format || !has_tile,
        "ScratchpadSpec '{}' has tile_format_metadata but no data_format_metadata",
        scratchpad.unique_id);
    if (has_format) {
        TT_FATAL(
            tt::is_data_format_supported(scratchpad.data_format_metadata.value(), arch),
            "ScratchpadSpec '{}' has data format '{}' which is not supported on architecture {}",
            scratchpad.unique_id,
            scratchpad.data_format_metadata.value(),
            arch);
    }

    if (const auto& borrowing = scratchpad.advanced_options.borrowed_from) {
        NodeRangeSet occupied_nodes;
        if (const auto it = collected.scratchpad_binders.find(scratchpad.unique_id);
            it != collected.scratchpad_binders.end()) {
            for (const auto* kernel : it->second) {
                occupied_nodes = occupied_nodes.merge(collected.kernel_node_set.at(kernel->unique_id));
            }
        }
        ValidateSRAMArenaBorrowing(
            SRAMArenaSource{.allocation = borrowing->allocation, .offset = borrowing->offset},
            collected,
            fmt::format("ScratchpadSpec '{}'", scratchpad.unique_id),
            occupied_nodes,
            scratchpad.size_per_node,
            l1_alignment);
    }
}

void ValidateScratchpadBindings(const KernelSpec& kernel) {
    std::unordered_set<std::string> accessor_names;
    for (const auto& binding : kernel.scratchpad_bindings) {
        auto [it, inserted] = accessor_names.insert(binding.accessor_name);
        TT_FATAL(
            inserted,
            "Kernel '{}' has duplicate scratchpad accessor_name '{}'",
            kernel.unique_id,
            binding.accessor_name);
        TT_FATAL(
            IsValidCppIdentifier(binding.accessor_name),
            "Kernel '{}' scratchpad accessor_name '{}' must be a valid C++ identifier",
            kernel.unique_id,
            binding.accessor_name);
        ValidateAccessorNameLength(kernel.unique_id, "scratchpad", binding.accessor_name);
    }
}

void ValidateScratchpadsUsed(const ValidationContext& ctx) {
    const ProgramSpec& spec = ctx.spec;
    const CollectedSpecData& collected = ctx.collected;

    // Every declared scratchpad must be bound by some kernel: an unbound scratchpad would reserve L1
    // that no kernel can reach.
    for (const auto& scratchpad : spec.scratchpads) {
        TT_FATAL(
            collected.scratchpad_binders.contains(scratchpad.unique_id),
            "ScratchpadSpec '{}' is declared but not bound by any kernel.",
            scratchpad.unique_id);
    }
}

}  // namespace tt::tt_metal::experimental
