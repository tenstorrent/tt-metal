// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>

namespace tt::tt_metal::experimental {

// Merges the allocator states from the given source allocators into |target|,
// replacing its current state. This is a convenience wrapper around
// Allocator::extract_state(), AllocatorState::merge(), and
// Allocator::override_state() that avoids exposing AllocatorState to callers.
//
// This API is experimental and may change or be removed without notice.
void synchronize_allocator_state(Allocator* target, const std::vector<Allocator*>& sources);

// Lowest address a program's program-local L1 (static circular buffers, dataflow buffers, scratchpads) can take on
// |cores|: the persistent L1 arena's high-water mark there. A program's first static circular buffer on exactly these
// cores is placed at this address.
//
// This API is experimental and may change or be removed without notice.
DeviceAddr program_local_l1_base(IDevice* device, const CoreRangeSet& cores);

}  // namespace tt::tt_metal::experimental
