// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
 * | device         | The device to create the semaphore on                  | IDevice*                                                  |              | Yes      |
 * | cores          | Range of the Tensix co-ordinates using the semaphore   | const CoreRangeSet &                                      |              | Yes      |
 * | initial_value  | Initial value of the semaphore                         | uint32_t                                                  |              | Yes      |
 * | buffer_type    | Buffer type to store the semaphore                     | BufferType                                                | L1 types     | No       |
 * | address        | Address of the semaphore to create                     | uint64_t                                                  |              | Yes      |
 */
// clang-format on
GlobalSemaphore CreateGlobalSemaphore(
    IDevice* device,
    const CoreRangeSet& cores,
    std::optional<uint32_t> initial_value,
    BufferType buffer_type,
    uint64_t address);
}  // namespace tt::tt_metal::experimental
