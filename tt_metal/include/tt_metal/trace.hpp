// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

namespace tt::tt_metal{
namespace v1 {

//==================================================
//                  TRACE OPERATIONS
//==================================================

/**
 * @brief Begins capture on a trace. Captured commands will have their execution delayed until the trace is replayed.
 *
 * @param device The device being traced.
 * @param cq The command queue associated with the trace.
 * @return Trace ID.
 */
uint32_t BeginTraceCapture(Device device, CommandQueue cq);

/**
 * @brief Ends capture on a trace. The trace can be replayed on the same device command queue.
 *
 * @param device The device being traced.
 * @param cq The command queue associated with the trace.
 * @param tid The trace ID returned by BeginTraceCapture.
 */
void EndTraceCapture(Device device, CommandQueue cq, uint32_t tid);

/**
 * @brief Replays a captured trace on the device.
 *
 * @param device The device holding the trace.
 * @param cq The command queue associated with the trace.
 * @param trace The trace ID to replay.
 * @param blocking Indicates whether the operation is blocking.
 */
void ReplayTrace(Device device, CommandQueue cq, Trace trace, bool blocking);

/**
 * @brief Releases a previously captured trace, deallocating associated resources.
 *
 * @param device The device holding the trace.
 * @param trace The trace to release.
 */
void ReleaseTrace(Device device, Trace trace);

/**
 * @brief Enqueues a trace for execution on the device.
 *
 * @param cq The command queue used to dispatch the trace.
 * @param trace The Trace to enqueue.
 * @param blocking Indicates whether the operation is blocking.
 */
void EnqueueTrace(CommandQueue cq, Trace trace, bool blocking);


} // namespace v1
} // namespace tt::tt_metal
