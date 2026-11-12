// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "impl/buffers/tensor_prefetcher_manager.hpp"

namespace tt::tt_metal::distributed {

// Tensor prefetcher internals that only tests use, kept out of TensorPrefetcherManager's API.
class TensorPrefetcherManagerTestAccess {
public:
    // What StartTensorPrefetcher requires of `mesh_device`. IsTensorPrefetcherSupported is this. It accepts a
    // sender core that is running a Program, so it holds while receiver Programs and prefetcher sessions run.
    static TensorPrefetcherPreconditions preconditions(const MeshDevice& mesh_device);

    // Records `prefetcher_pipes` on their senders for Stop's DRAIN, as queueing a request to them would, without
    // sending a request. For tests that drain live PrefetcherPipes without the PREFETCH path. Same pipe
    // preconditions as TensorPrefetcherManager::queue.
    static void register_drain_target(
        TensorPrefetcherManager& manager,
        const std::vector<std::reference_wrapper<const experimental::PrefetcherPipe>>& prefetcher_pipes);

    // A CQ fence wait the dispatcher never satisfies: the local DRISC L1 address of the CQ signal slot, the same on
    // every provisioned sender, and the value the senders wait for there.
    struct UnsignaledCqWait {
        uint32_t slot_address = 0;
        uint32_t value = 0;
    };

    // enqueue_cq_signal_and_wait for command queue 0 on every device, without the dispatcher write: it bumps the
    // counter and queues the WAIT_SIGNAL on CQ fence slot 0, so the senders wait until the caller writes `value` into
    // the slot. For tests of the fence, which slow dispatch otherwise satisfies before the wait is queued.
    static UnsignaledCqWait queue_cq_wait_without_signal(TensorPrefetcherManager& manager);
};

}  // namespace tt::tt_metal::distributed
