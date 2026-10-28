// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <variant>

#include "impl/metal2_host_api/program_spec/validation/resource/resource.hpp"

namespace tt::tt_metal::experimental {

void ValidateRawAllocationParameter(const RawAllocationParameter& parameter) {
    const auto per_bank_size =
        std::visit([](const auto& properties) { return properties.per_bank_size; }, parameter.properties);
    TT_FATAL(
        per_bank_size > 0,
        "RawAllocationParameter '{}' has per_bank_size = 0; per_bank_size must be greater than zero.",
        parameter.unique_id);
    if (const auto* range = std::get_if<RangeLockstepAllocationProperties>(&parameter.properties)) {
        TT_FATAL(
            !range->banks_to_be_allocated_on.empty(),
            "RawAllocationParameter '{}' has empty banks_to_be_allocated_on; at least one node is required.",
            parameter.unique_id);
    }
}

void ValidateRawAllocationBorrowing(
    const RawAllocationSource& source,
    const CollectedSpecData& collected,
    std::string_view borrower,
    const NodeRangeSet& occupied_nodes,
    uint64_t size_per_node,
    uint32_t l1_alignment) {
    const auto it = collected.raw_allocation_parameter_by_name.find(source.allocation);
    TT_FATAL(
        it != collected.raw_allocation_parameter_by_name.end(),
        "{} borrows memory from unknown RawAllocationParameter '{}'.",
        borrower,
        source.allocation);
    TT_FATAL(
        source.offset % l1_alignment == 0,
        "{} borrowing offset {} must be a multiple of L1 alignment {}.",
        borrower,
        source.offset,
        l1_alignment);

    const auto& properties = it->second->properties;
    if (const auto* range = std::get_if<RangeLockstepAllocationProperties>(&properties)) {
        TT_FATAL(
            range->banks_to_be_allocated_on.contains(occupied_nodes),
            "{} borrows from RawAllocationParameter '{}', which does not cover every occupied node.",
            borrower,
            source.allocation);
    }
    ValidateRawAllocationCapacity(source, properties, borrower, size_per_node);
}

}  // namespace tt::tt_metal::experimental
