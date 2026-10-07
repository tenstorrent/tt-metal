// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>

namespace tt::tt_metal::experimental::range_lockstep_allocation {

// Experimental and subject to change: this header carries no API-stability guarantee.
//
// Allocates a global semaphore whose L1 reservation is scoped to `cores` (range lockstep) instead
// of being kept clear of per-core allocations on every core. The semaphore still takes one address
// across `cores`, so a single compile-time address remains valid on every participant.
//
// Use it only when nothing signals or reads the semaphore on a core outside `cores`; a remote
// increment on any other core may land on top of a per-core allocation there.
GlobalSemaphore create_global_semaphore(
    distributed::MeshDevice& device,
    const CoreRangeSet& cores,
    uint32_t initial_value,
    BufferType buffer_type = BufferType::L1);

}  // namespace tt::tt_metal::experimental::range_lockstep_allocation
