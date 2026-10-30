// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal {
class Program;
}  // namespace tt::tt_metal

// Experimental and subject to change: this header carries no API-stability guarantee.
namespace tt::tt_metal::experimental {

/// The logical cores of `core_type` on which `program` runs at least one kernel. For Tensix and active Ethernet cores,
/// these are the cores dispatch assigns the program's sub-device from.
CoreRangeSet GetKernelCores(const Program& program, HalProgrammableCoreType core_type);

}  // namespace tt::tt_metal::experimental
