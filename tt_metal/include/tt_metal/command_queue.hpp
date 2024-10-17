// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

#include "tt_metal/impl/buffers/buffer.hpp"

//==================================================
//                COMMAND QUEUE OPERATIONS
//==================================================


namespace tt::tt_metal{
namespace v1 {

/**
 * @brief Reads a buffer from the device.
 *
 * @param cq The command queue used to dispatch the command.
 * @param buffer The device buffer to read from.
 * @param dst Pointer to the destination memory where data will be stored.
 * @param blocking Indicates whether the operation is blocking.
 */
void EnqueueReadBuffer(
    CommandQueue cq,
    Buffer buffer,
    std::byte *dst,
    bool blocking);

/**
 * @brief Writes data to a buffer on the device.
 *
 * @param cq The command queue used to dispatch the command.
 * @param buffer The device buffer to write to.
 * @param src Source data vector to write to the device.
 * @param blocking Indicates whether the operation is blocking.
 */
void EnqueueWriteBuffer(
    CommandQueue cq,
    Buffer buffer,
    const std::byte *src,
    bool blocking);


/**
 * @brief Writes a program to the device and launches it.
 *
 * @param cq The command queue used to dispatch the command.
 * @param program The program to execute on the device.
 * @param blocking Indicates whether the operation is blocking.
 */
void EnqueueProgram(CommandQueue cq, Program program, bool blocking);

/**
 * @brief Blocks until all previously dispatched commands on the device have completed.
 *
 * @param cq The command queue to wait on.
 */
void Finish(CommandQueue cq);


/**
 * @brief Sets the command queue mode to lazy or immediate.
 *
 * @param lazy If true, sets the command queue to lazy mode.
 */
void SetLazyCommandQueueMode(bool lazy);


/**
 * @brief Retrieves the device associated with the command queue.
 *
 * @param cq The command queue to query.
 * @return Device handle associated with the command queue.
 */
Device GetDevice(class CommandQueue cq);

/**
 * @brief Retrieves the ID of the command queue.
 *
 * @param cq The command queue to query.
 * @return ID of the command queue.
 */
uint32_t GetId(class CommandQueue cq);

} // namespace v1
} // namespace tt::tt_metal
