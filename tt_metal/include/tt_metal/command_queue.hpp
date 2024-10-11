// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/tt_stl/span.hpp"
#include "types.hpp"

//==================================================
//                COMMAND QUEUE OPERATIONS
//==================================================

namespace tt::tt_metal {
namespace v1 {

/**
 * @brief Reads a buffer from the device.
 *
 * @param cq The command queue used to dispatch the command.
 * @param buffer The device buffer to read from.
 * @param dst Destination memory to copy from device to host.
 * @param blocking Indicates whether the operation is blocking.
 */
void EnqueueReadBuffer(CommandQueueHandle cq, Buffer buffer, stl::Span<std::byte> dst, bool blocking);

/**
 * @brief Writes data to a buffer on the device.
 *
 * @param cq The command queue used to dispatch the command.
 * @param buffer The device buffer to write to.
 * @param src Source memory to copy from host to device.
 * @param blocking Indicates whether the operation is blocking.
 */
void EnqueueWriteBuffer(CommandQueueHandle cq, Buffer buffer, stl::Span<const std::byte> src, bool blocking);

/**
 * @brief Writes a program to the device and launches it.
 *
 * @param cq The command queue used to dispatch the command.
 * @param program The program to execute on the device.
 * @param blocking Indicates whether the operation is blocking.
 */
void EnqueueProgram(CommandQueueHandle cq, ProgramHandle program, bool blocking);

/**
 * @brief Blocks until all previously dispatched commands on the device have completed.
 *
 * @param cq The command queue to wait on.
 */
void Finish(CommandQueueHandle cq);

/**
 * @brief Sets the command queue mode to lazy or immediate.
 *
 * @param cq The command queue to set mode on
 * @param mode What mode to set the command queue
 */
void SetCommandQueueMode(CommandQueueHandle cq, CommandQueueMode mode);

/**
 * @brief Retrieves the device associated with the command queue.
 *
 * @param cq The command queue to query.
 * @return Device handle associated with the command queue.
 */
DeviceHandle GetDevice(CommandQueueHandle cq);

/**
 * @brief Retrieves the ID of the command queue.
 *
 * @param cq The command queue to query.
 * @return ID of the command queue.
 */
std::uint32_t GetId(CommandQueueHandle cq);

}  // namespace v1
}  // namespace tt::tt_metal
