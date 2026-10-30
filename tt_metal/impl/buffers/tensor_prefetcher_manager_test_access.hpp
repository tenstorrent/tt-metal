// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "impl/buffers/tensor_prefetcher_manager.hpp"

namespace tt::tt_metal::distributed {

// Tensor prefetcher internals that only tests use, kept out of TensorPrefetcherManager's API.
class TensorPrefetcherManagerTestAccess {
public:
    // What StartTensorPrefetcher requires of `mesh_device`. IsTensorPrefetcherSupported is this. On Quasar
    // it accepts a sender CCE whose GO_MSG holds RUN_MSG_GO, so it holds while receiver Programs and
    // prefetcher sessions run.
    static TensorPrefetcherPreconditions preconditions(const MeshDevice& mesh_device);

    // TL1 offset of every Quasar CCE sender's control block (tensor_prefetcher_cce_abi.hpp). It depends only on
    // the mesh's DRISC L1 arena, so it holds before Start and after Stop.
    static uint32_t cce_sender_control_base(const MeshDevice& mesh_device);

    // Records `gcb` on its senders for Stop's DRAIN, as queueing a request to it would, without sending a
    // request. For tests that drain a live GlobalCircularBuffer without the PREFETCH path.
    static void register_drain_target(TensorPrefetcherManager& manager, const experimental::GlobalCircularBuffer& gcb);

    // A WAIT_CQ the dispatcher never satisfies: the local DRISC L1 address of the CQ signal slot, the same on every
    // provisioned sender, and the value the senders wait for there.
    struct UnsignaledCqWait {
        uint32_t slot_address = 0;
        uint32_t value = 0;
    };

    // enqueue_cq_signal_and_wait for command queue 0 on every device, without the dispatcher write: it bumps the
    // counter and queues the WAIT_CQ, so the senders wait until the caller writes `value` into the slot. For tests of
    // the fence, which slow dispatch otherwise satisfies before the WAIT_CQ is queued.
    static UnsignaledCqWait queue_cq_wait_without_signal(TensorPrefetcherManager& manager);
};

}  // namespace tt::tt_metal::distributed
