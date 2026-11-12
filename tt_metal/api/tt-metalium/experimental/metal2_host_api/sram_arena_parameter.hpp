// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <tt-metalium/experimental/sram_arena/sram_arena_properties.hpp>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::experimental {

// ============================================================================
//  SRAMArenaParameter API
// ============================================================================
//
// An SRAMArenaParameter is used to declare that a Program operates on a
// user-managed SRAMArena object with defined properties.
//
// A matching SRAMArena object will be supplied at Program execution time
// via ProgramRunArgs. If its properties do not match the declared
// SRAMArenaParameter, a runtime error is triggered.
//
// ============================================================================

// A name identifying an SRAMArenaParameter within a ProgramSpec.
using SRAMArenaParamName = ttsl::StrongType<std::string, struct SRAMArenaParamNameTag>;

struct SRAMArenaParameter {
    // Allocation identifier: used to reference this SRAMArena within the ProgramSpec
    SRAMArenaParamName unique_id;

    // SRAMArena properties (size, node range, etc)
    tt::tt_metal::experimental::SRAMArenaProperties properties;
};

}  // namespace tt::tt_metal::experimental
