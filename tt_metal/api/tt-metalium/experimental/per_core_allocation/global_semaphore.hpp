// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>

namespace tt::tt_metal::experimental::per_core_allocation {

// Experimental and subject to change: this header carries no API-stability guarantee.
//
// Allocates an L1 global semaphore whose address is reserved only on `cores`. The semaphore takes
// one address on every core of `cores` and every device of the mesh, so a single compile-time
// address stays valid on every participant, but each core reserves it in its own per-core
// allocator (see set_uniform_address in buffer.hpp). Cores outside `cores` can place other buffers
// at that address.
//
// Use it only when nothing signals or reads the semaphore on a core outside `cores`, including
// through a multicast whose rectangle reaches past them: a write there lands on whatever that core
// has allocated at the address.
//
// Needs the HYBRID allocator; with any other mode the semaphore is reserved on every core, as
// GlobalSemaphore's constructor does.
GlobalSemaphore create_global_semaphore(
    distributed::MeshDevice& device, const CoreRangeSet& cores, uint32_t initial_value);

}  // namespace tt::tt_metal::experimental::per_core_allocation
