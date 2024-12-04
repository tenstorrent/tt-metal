// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

namespace tt::tt_metal {
namespace v1 {

//==================================================
//                  TRACE OPERATIONS
//==================================================

/**
 * @brief Obtains the associated command queue from a given trace handle.
 *
 * @param trace The trace handle returned by BeginTraceCapture.
 * @return Command queue handle.
 */
CommandQueueHandle GetCommandQueue(TraceHandle trace);

/**
 * @brief Begins capture on a trace. Captured commands will have their execution delayed until the trace is replayed.
 *
 * @param cq The command queue associated with the trace.
 * @return Trace handle.
 */
TraceHandle BeginTraceCapture(CommandQueueHandle cq);

/**
 * @brief Ends capture on a trace. The trace can be replayed on the same device command queue.
 *
 * @param trace The trace handle returned by BeginTraceCapture.
 */
void EndTraceCapture(TraceHandle trace);

/**
 * @brief Replays a captured trace on the device.
 *
 * @param trace The trace to replay.
 * @param blocking Indicates whether the operation is blocking.
 */
void ReplayTrace(TraceHandle trace, bool blocking);

/**
 * @brief Releases a previously captured trace, deallocating associated resources.
 *
 * @param trace The trace to release.
 */
void ReleaseTrace(TraceHandle trace);

/**
 * @brief Enqueues a trace for execution on the device.
 *
 * @param trace The Trace to enqueue.
 * @param blocking Indicates whether the operation is blocking.
 */
void EnqueueTrace(TraceHandle trace, bool blocking);

}  // namespace v1
}  // namespace tt::tt_metal
