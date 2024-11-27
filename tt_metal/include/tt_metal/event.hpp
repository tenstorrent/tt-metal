// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

//==================================================
//                  EVENT MANAGEMENT
//==================================================

namespace tt::tt_metal {
namespace v1 {

/**
 * @brief Enqueues a command to record an event on the device.
 *
 * @param cq The command queue used to dispatch the command.
 * @return Handle to the recorded Event object.
 */
EventHandle EnqueueRecordEvent(CommandQueueHandle cq);

/**
 * @brief Enqueues a command to wait for an event to complete on the device.
 *
 * @param cq The command queue that will wait for the event.
 * @param event Handle to the Event object to wait on.
 */
void EnqueueWaitForEvent(CommandQueueHandle cq, const EventHandle& event);

/**
 * @brief Blocks the host until the specified event has completed on the device.
 *
 * @param event Handle to the Event object to synchronize.
 */
void EventSynchronize(const EventHandle& event);

/**
 * @brief Queries the completion status of an event on the device.
 *
 * @param event Handle to the Event object to query.
 * @return True if the event is completed; otherwise, false.
 */
bool EventQuery(const EventHandle& event);

/**
 * @brief Synchronizes the device with the host by waiting for all operations to complete.
 *
 * @param device device to synchronize.
 */
void DeviceSynchronize(DeviceHandle device);

/**
 * @brief Synchronizes the command queue with the host by waiting for all operations to complete.
 *
 * @param cq command queue to synchronize.
 */
void CommandQueueSynchronize(CommandQueueHandle cq);

}  // namespace v1
}  // namespace tt::tt_metal
