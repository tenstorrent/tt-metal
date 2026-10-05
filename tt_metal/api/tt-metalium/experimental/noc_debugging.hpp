// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace tt::tt_metal::experimental {

struct NocDebugStateSummary {
    bool enabled = false;
    bool collector_ready = false;
    bool includes_dispatch_cores = false;
    std::size_t issues = 0;
    std::size_t unflushed_atomic_issues = 0;
    std::size_t observed_atomic_events = 0;
    std::size_t pending_events = 0;
};

// These test-support functions read or reset the NoC debug state for the complete Metal context. The caller must
// synchronize all devices and must have exclusive ownership of every mesh in that context.
NocDebugStateSummary GetNocDebugStateSummary(distributed::MeshDevice& mesh_device);

void ResetNocDebugState(distributed::MeshDevice& mesh_device);

}  // namespace tt::tt_metal::experimental
