// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

//==================================================
//                  EVENT MANAGEMENT
//==================================================

namespace tt::tt_metal{
namespace v1 {

/**
 * @brief Enqueues a command to record an event on the device.
 *
 * @param cq The command queue used to dispatch the command.
 * @param event Shared pointer to the Event object to record.
 */
void EnqueueRecordEvent(CommandQueue cq, const std::shared_ptr<Event> &event);

/**
 * @brief Enqueues a command to wait for an event to complete on the device.
 *
 * @param cq The command queue that will wait for the event.
 * @param event Shared pointer to the Event object to wait on.
 */
void EnqueueWaitForEvent(CommandQueue cq, const std::shared_ptr<Event> &event);

/**
 * @brief Blocks the host until the specified event has completed on the device.
 *
 * @param event Shared pointer to the Event object to synchronize.
 */
void EventSynchronize(const std::shared_ptr<Event> &event);

/**
 * @brief Queries the completion status of an event on the device.
 *
 * @param event Shared pointer to the Event object to query.
 * @return True if the event is completed; otherwise, false.
 */
bool EventQuery(const std::shared_ptr<Event> &event);


/**
 * @brief Synchronizes the device with the host by waiting for all operations to complete.
 *
 * @param device The device to synchronize.
 * @param cq_id Optional command queue ID to synchronize. If not provided, all queues are synchronized.
 */
void Synchronize(Device device, const std::optional<uint8_t> cq_id = std::nullopt);


} // namespace v1
} // namespace tt::tt_metal
