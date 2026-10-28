// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/metal2_host_api/raw_allocation_support.hpp"

#include <variant>

#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>

namespace tt::tt_metal {

void ValidateRawAllocationCapacity(
    const RawAllocationSource& source,
    const experimental::RawAllocationProperties& properties,
    std::string_view borrower,
    uint64_t size_per_node) {
    const auto per_bank_size = std::visit([](const auto& props) { return props.per_bank_size; }, properties);
    // Widen before adding the offset: the requested extent, not allocator padding, is the limit.
    TT_FATAL(
        uint64_t{source.offset} + size_per_node <= per_bank_size,
        "{} borrowing offset {} + size_per_node {} exceeds RawAllocationParameter '{}' per_bank_size {}.",
        borrower,
        source.offset,
        size_per_node,
        source.allocation,
        per_bank_size);
}

}  // namespace tt::tt_metal
