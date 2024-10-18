// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

#include "tt_metal/impl/buffers/buffer.hpp"

//==================================================
//                COMMAND QUEUE OPERATIONS
//==================================================

namespace tt::tt_metal {

inline namespace v0 {

class Program;
class Event;

}

namespace v1 {

/**
 * @brief Get a handle to a command queue on a device.
 *
 * @param device The device to get the command queue from.
 * @param id The id of the command queue to get.
 * @return A handle to the command queue.
 */
CommandQueueHandle GetCommandQueue(DeviceHandle device, uint32_t id = 0);

/**
 * @brief Reads a buffer from the device.
 *
 * @param cq The command queue used to dispatch the command.
 * @param buffer The device buffer to read from.
 * @param dst The vector where the results that are read will be stored.
 * @param blocking Indicates whether the operation is blocking.
 */
void EnqueueReadBuffer(
    CommandQueueHandle cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    std::vector<uint32_t> &dst,
    bool blocking);

/**
 * @brief Reads a buffer from the device.
 *
 * @param cq The command queue used to dispatch the command.
 * @param buffer The device buffer to read from.
 * @param dst Pointer to the destination memory where data will be stored.
 * @param blocking Indicates whether the operation is blocking.
 */
void EnqueueReadBuffer(
    CommandQueueHandle cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    void *dst,
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
    CommandQueueHandle cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    std::vector<uint32_t> &src,
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
    CommandQueueHandle cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    HostDataType src,
    bool blocking);

/**
 * @brief Writes a program to the device and launches it.
 *
 * @param cq The command queue used to dispatch the command.
 * @param program The program to execute on the device.
 * @param blocking Indicates whether the operation is blocking.
 */
void EnqueueProgram(CommandQueueHandle cq, v0::Program& program, bool blocking);

/**
 * @brief Blocks until all previously dispatched commands on the device have completed.
 *
 * @param cq The command queue to wait on.
 */
void Finish(CommandQueueHandle cq);

/**
 * @brief Enqueues a trace of previously generated commands and data.
 *
 * @param cq The command queue used to dispatch the command.
 * @param trace_id A unique id representing an existing on-device trace, which has been
 * instantiated via InstantiateTrace where the trace_id is returned
 * @param blocking Indicates whether the operation is blocking.
 */
void EnqueueTrace(CommandQueueHandle cq, uint32_t trace_id, bool blocking);

/**
 * @brief Enqueues a command to record an Event on the device for a given CQ, and updates the Event object for the user.
 *
 * @param cq The command queue used to dispatch the command.
 * @param event An event that will be populated by this function, and inserted in CQ
 */
void EnqueueRecordEvent(CommandQueueHandle cq, const std::shared_ptr<Event> &event);

/**
 * @brief Enqueues a command on the device for a given CQ (non-blocking). The command on device will block and wait for completion of the specified event (which may be in another CQ).
 *
 * @param cq The command queue used to dispatch the command.
 * @param event The event object that this CQ will wait on for completion.
 */
void EnqueueWaitForEvent(CommandQueueHandle cq, const std::shared_ptr<Event> &event);

} // namespace v1

} // namespace tt::tt_metal
