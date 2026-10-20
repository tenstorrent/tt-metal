// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <tt-metalium/experimental/allocation/raw_allocation_properties.hpp>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::experimental {

// ============================================================================
//  RawAllocationParameter API
// ============================================================================
//
// A RawAllocationParameter is used to declare that a Program operates on a
// user-managed RawAllocation object with defined properties.
//
// A matching RawAllocation object will be supplied at Program execution time
// via ProgramRunArgs. If its properties do not match the declared
// RawAllocationParameter, a runtime error is triggered.
//
// ============================================================================

// A name identifying a RawAllocationParameter within a ProgramSpec.
using RawAllocationParamName = ttsl::StrongType<std::string, struct RawAllocationParamNameTag>;

struct RawAllocationParameter {
    // Allocation identifier: used to reference this RawAllocation within the ProgramSpec
    RawAllocationParamName unique_id;

    // RawAllocation properties (size, node range, etc)
    tt::tt_metal::experimental::RawAllocationProperties properties;
};

}  // namespace tt::tt_metal::experimental
