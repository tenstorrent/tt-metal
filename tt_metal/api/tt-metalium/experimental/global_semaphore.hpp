// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>

namespace tt::tt_metal::experimental {

// clang-format off
/**
 * Creates a global semaphore in the PersistentL1Arena: 4 bytes per core, at the same address on every
 * core in `cores` and on every device of `device`. The space is freed when the last copy of the
 * returned GlobalSemaphore is destroyed.
 *
 * Must be called before any program places program-local L1 on `cores`; allocation fails otherwise.
 * reset_semaphore_value() writes through the command queue, ordered after previously enqueued programs.
 *
 * Return value: GlobalSemaphore
 *
 * | Argument       | Description                                            | Type                     | Required |
 * |----------------|--------------------------------------------------------|--------------------------|----------|
 * | device         | The mesh device to create the semaphore on             | distributed::MeshDevice& | Yes      |
 * | cores          | Logical worker cores that hold a copy of the semaphore | const CoreRangeSet&      | Yes      |
 * | initial_value  | Initial value of the semaphore                         | uint32_t                 | Yes      |
 */
// clang-format on
GlobalSemaphore CreatePersistentGlobalSemaphore(
    distributed::MeshDevice& device, const CoreRangeSet& cores, uint32_t initial_value);

}  // namespace tt::tt_metal::experimental
