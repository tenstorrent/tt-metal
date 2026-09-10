// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/global_semaphore.hpp>

namespace tt::tt_metal::experimental {
// clang-format off
/**
 * Experimental API for creating a global semaphore at a specific address.
 * and with optional initial value used by tt-mlir.
 *
 * Return value: GlobalSemaphore
 *
 * | Argument       | Description                                            | Type                                                      | Valid Range  | Required |
 * |----------------|--------------------------------------------------------|-----------------------------------------------------------|--------------|----------|
 * | device         | The mesh device to create the semaphore on             | distributed::MeshDevice&                                  |              | Yes      |
 * | cores          | Range of the Tensix coordinates using the semaphore    | const CoreRangeSet &                                      |              | Yes      |
 * | initial_value  | Initial value of the semaphore                         | uint32_t                                                  |              | Yes      |
 * | buffer_type    | Buffer type to store the semaphore                     | BufferType                                                | L1 types     | No       |
 * | address        | Address of the semaphore to create                     | uint64_t                                                  |              | Yes      |
 */
// clang-format on
GlobalSemaphore CreateGlobalSemaphore(
    distributed::MeshDevice& device,
    const CoreRangeSet& cores,
    std::optional<uint32_t> initial_value,
    BufferType buffer_type,
    uint64_t address);

[[deprecated(
    "Use CreateGlobalSemaphore(distributed::MeshDevice&, ...) instead. "
    "CreateGlobalSemaphore(IDevice*, ...) will be removed after 2026-09-20.")]]
GlobalSemaphore CreateGlobalSemaphore(
    IDevice* device,
    const CoreRangeSet& cores,
    std::optional<uint32_t> initial_value,
    BufferType buffer_type,
    uint64_t address);
}  // namespace tt::tt_metal::experimental
