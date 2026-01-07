// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

// Update shared memory statistics from device allocators
// This queries each device's allocator for actual current memory usage
// and updates the shared memory region that tt-smi reads.
//
// Call this periodically or before querying stats to ensure accuracy.
void UpdateAllocatorStatsToShm();

}  // namespace tt::tt_metal
