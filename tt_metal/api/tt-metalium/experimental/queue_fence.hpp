// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/sub_device_types.hpp>

namespace tt::tt_metal::distributed {
class MeshCommandQueue;
}  // namespace tt::tt_metal::distributed

// Experimental and subject to change: this header carries no API-stability guarantee.
namespace tt::tt_metal::experimental {

/// A point in one command queue's program order on one sub-device, or in its host transfers. Taking it enqueues
/// nothing. It completes once every program the queue had enqueued on that sub-device by then has finished, or, for a
/// transfer fence, once the queue has carried out the transfers it had been given. A queue does not order the programs
/// it runs on different sub-devices, so a program fence names one sub-device.
struct QueueFence {
    uint32_t cq_id = 0;
    // Nothing for a transfer fence.
    std::optional<SubDeviceId> sub_device_id;
    // The queue restarts a sub-device's worker counter behind a wait for its earlier programs (a counter wrap, a trace
    // replay, a sub-device manager change, a clear). Each restart begins a new epoch. A fence from an earlier epoch is
    // ordered before every later command of its queue; that does not mean the host knows it has completed.
    uint64_t epoch = 0;
    // The sub-device's worker-completion count once the fence's programs have finished.
    uint32_t count = 0;
};

/// The fence after everything `cq` has enqueued so far on `sub_device_id`.
QueueFence GetQueueFence(distributed::MeshCommandQueue& cq, SubDeviceId sub_device_id);

/// The fence after the host transfers (buffer writes and reads) `cq` has been given so far. Waiting for it from another
/// queue also waits for the transfers given to `cq` later, but not for its programs.
QueueFence GetQueueTransferFence(distributed::MeshCommandQueue& cq);

/// Makes `cq` run nothing enqueued after this call until `fence` completes; later programs on the fence's queue are
/// not waited for. On the fence's own queue the dispatcher waits on the sub-device's worker counter. On another queue,
/// an event covering exactly the fence's programs is recorded on the fence's queue, and `cq` waits for it.
void EnqueueWaitForQueueFence(distributed::MeshCommandQueue& cq, const QueueFence& fence);

}  // namespace tt::tt_metal::experimental
