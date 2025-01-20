// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

//==================================================
//                COMMAND QUEUE OPERATIONS
//==================================================

namespace tt::tt_metal {
namespace v1 {

/**
 * @brief Retrieves a command queue from the device for a given queue ID.
 *
 * @param device The device to query.
 * @param cq_id The command queue ID.
 * @return CommandQueue handle.
 */
CommandQueueHandle GetCommandQueue(IDevice* device, std::uint8_t cq_id);

/**
 * @brief Retrieves the default command queue for the given device.
 *
 * @param device The device to query.
 * @return CommandQueue handle.
 */
CommandQueueHandle GetDefaultCommandQueue(IDevice* device);

/**
 * @brief Reads a buffer from the device.
 *
 * @param cq The command queue used to dispatch the command.
 * @param buffer The device buffer to read from.
 * @param dst Pointer to the destination memory where data will be stored.
 * @param blocking Indicates whether the operation is blocking.
 */
void EnqueueReadBuffer(CommandQueueHandle cq, const BufferHandle& buffer, std::byte* dst, bool blocking);

/**
 * @brief Writes data to a buffer on the device.
 *
 * @param cq The command queue used to dispatch the command.
 * @param buffer The device buffer to write to.
 * @param src Source data vector to write to the device.
 * @param blocking Indicates whether the operation is blocking.
 */
void EnqueueWriteBuffer(CommandQueueHandle cq, const BufferHandle& buffer, const std::byte* src, bool blocking);

/**
 * @brief Writes a program to the device and launches it.
 *
 * @param cq The command queue used to dispatch the command.
 * @param program The program to execute on the device.
 * @param blocking Indicates whether the operation is blocking.
 */
void EnqueueProgram(CommandQueueHandle cq, ProgramHandle& program, bool blocking);

/**
 * @brief Blocks until all previously dispatched commands on the device have completed.
 *
 * @param cq The command queue to wait on.
 * @param sub_device_ids The sub-device ids to wait for completion on. If empty, waits for all sub-devices.
 */
void Finish(CommandQueueHandle cq, tt::stl::Span<const SubDeviceId> sub_device_ids = {});

/**
 * @brief Retrieves the device associated with the command queue.
 *
 * @param cq The command queue to query.
 * @return Device handle associated with the command queue.
 */
IDevice* GetDevice(CommandQueueHandle cq);

/**
 * @brief Retrieves the ID of the command queue.
 *
 * @param cq The command queue to query.
 * @return ID of the command queue.
 */
std::uint8_t GetId(CommandQueueHandle cq);

}  // namespace v1
}  // namespace tt::tt_metal
