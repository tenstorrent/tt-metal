// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/allocator.hpp>

namespace tt::tt_metal::experimental {

// Merges the allocator states from the given source allocators into |target|,
// replacing its current state. This is a convenience wrapper around
// Allocator::extract_state(), AllocatorState::merge(), and
// Allocator::override_state() that avoids exposing AllocatorState to callers.
//
// This API is experimental and may change or be removed without notice.
void synchronize_allocator_state(Allocator* target, const std::vector<Allocator*>& sources);

// Reserves in |target| every region |source| has allocated (DRAM, L1, L1_SMALL, TRACE) that |target| has free,
// leaving |target|'s own allocations intact (unlike synchronize_allocator_state, which coalesces them).
// For independent allocators over the same physical memory, e.g. a parent mesh and its unit submeshes.
// Returns the number of bytes reserved (per bank). Experimental.
DeviceAddr reserve_allocator_regions(Allocator* target, const Allocator* source);

}  // namespace tt::tt_metal::experimental
