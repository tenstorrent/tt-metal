// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

// Returns the list of logical Tensix cores reserved as fast-dispatch cores on user-facing chips.
// Useful for program factories that directly pick cores by physical layout (e.g. DRAM-bank
// adjacency) and need to avoid placing kernels on dispatch-reserved cores.
//
// The list is identical across user-facing chips and is stable for a given DispatchCoreConfig
// (in particular, it changes when the DispatchCoreAxis switches between ROW and COL).
const std::vector<CoreCoord>& get_logical_dispatch_cores_on_user_chips();

}  // namespace tt::tt_metal
