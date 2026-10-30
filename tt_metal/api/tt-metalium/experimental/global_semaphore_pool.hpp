// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

// clang-format off
/**
 * Reserves L1 for up to `num_slots` global semaphores on every worker core of the mesh device. Semaphores are
 * then created from it with CreateGlobalSemaphoreFromPool. Each one takes 16 bytes (one L1 alignment).
 *
 * Call it right after opening the device, before any program or L1 buffer is created on it. Otherwise it throws.
 * If it is never called, no L1 is reserved.
 *
 * A mesh and its submeshes share one pool, so calling it on either reserves it for both. Calling it again does
 * nothing if the pool already has at least `num_slots` slots, and throws if it would need to grow.
 * TT_METAL_GLOBAL_SEMAPHORE_POOL_SLOTS, if set, replaces `num_slots`.
 *
 * Return value: void
 *
 * | Argument       | Description                                            | Type                                                      | Valid Range  | Required |
 * |----------------|--------------------------------------------------------|-----------------------------------------------------------|--------------|----------|
 * | device         | The mesh device to reserve the pool on                 | distributed::MeshDevice&                                  |              | Yes      |
 * | num_slots      | Number of semaphores the pool can hold at once         | uint32_t                                                  | 0 = no pool  | Yes      |
 */
// clang-format on
void ReserveGlobalSemaphorePool(distributed::MeshDevice& device, uint32_t num_slots);

// clang-format off
/**
 * Creates a global semaphore from the mesh device's pool, with a value of 0. Throws if ReserveGlobalSemaphorePool
 * was not called or if the pool is full.
 *
 * Semaphores can be created at any time after the pool is reserved, even after programs have run on `cores`.
 * A semaphore has the same L1 address on every core and chip of the mesh. `cores` are the cores whose copy is
 * set by reset_semaphore_value().
 *
 * Copies of a GlobalSemaphore refer to the same semaphore. It goes back to the pool when the last copy is
 * destroyed, so keep a copy alive while any program or trace that uses it can still run.
 *
 * Return value: GlobalSemaphore
 *
 * | Argument       | Description                                            | Type                                                      | Valid Range  | Required |
 * |----------------|--------------------------------------------------------|-----------------------------------------------------------|--------------|----------|
 * | device         | The mesh device to create the semaphore on             | distributed::MeshDevice&                                  |              | Yes      |
 * | cores          | Tensix worker cores that use the semaphore             | const CoreRangeSet&                                       |              | Yes      |
 */
// clang-format on
GlobalSemaphore CreateGlobalSemaphoreFromPool(distributed::MeshDevice& device, const CoreRangeSet& cores);

}  // namespace experimental

}  // namespace tt::tt_metal
