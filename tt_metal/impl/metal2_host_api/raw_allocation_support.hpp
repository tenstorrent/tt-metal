// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string_view>

#include <tt-metalium/experimental/metal2_host_api/raw_allocation_parameter.hpp>

namespace tt::tt_metal {

// Identifies an allocation parameter and a byte offset within its reservation.
struct RawAllocationSource {
    experimental::RawAllocationParamName allocation;
    uint32_t offset = 0;
};

// Reject a borrower whose offset and per-node size exceed the declared allocation capacity.
void ValidateRawAllocationCapacity(
    const RawAllocationSource& source,
    const experimental::RawAllocationProperties& properties,
    std::string_view borrower,
    uint64_t size_per_node);

}  // namespace tt::tt_metal
