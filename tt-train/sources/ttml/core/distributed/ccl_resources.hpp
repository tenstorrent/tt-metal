// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <vector>

namespace ttml::core::distributed {

// Device-resident resources the collective wrappers in ttnn_fixed::distributed reuse across calls: the
// global semaphores.
//
// Collectives are cross-device operations. When device E starts a collective it writes into every peer's
// copy of the semaphores, while a slower peer D may still be finishing the previous collective on the same
// set. The semaphores therefore rotate through several sets, and the rotation depth is the slack that covers
// the skew between devices. Each command queue gets its own pool: collectives issued on different queues can
// be in flight at the same time (a sequence-parallel collective on the second queue next to a tensor-parallel
// all-reduce on the first, see ops/distributed/sp_overlap.hpp) and must not rotate through shared semaphores.
class CCLResources {
public:
    CCLResources();

    tt::tt_metal::GlobalSemaphore get_barrier_semaphore();
    std::vector<tt::tt_metal::GlobalSemaphore> get_all_gather_semaphore();
    std::vector<tt::tt_metal::GlobalSemaphore> get_reduce_scatter_semaphores();
    std::vector<tt::tt_metal::GlobalSemaphore> get_all_reduce_barrier_semaphores();

private:
    static constexpr uint32_t kNumSemaphoreSets = 8U;
    static constexpr uint32_t kNumSemaphoresPerAllGather = 2U;
    static constexpr uint32_t kNumSemaphoresPerReduceScatterCall = 3U;
    static constexpr uint32_t kNumSemaphoresPerAllReduceBarrierCall = 2U;

    // kNumSemaphoreSets groups of `per_call` semaphores handed out round-robin.
    struct SemaphoreRing {
        std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
        uint32_t per_call = 1U;
        uint32_t next = 0U;
        std::vector<tt::tt_metal::GlobalSemaphore> take();
    };
    struct Pool {
        SemaphoreRing barrier;
        SemaphoreRing all_gather;
        SemaphoreRing reduce_scatter;
        SemaphoreRing all_reduce_barrier;
    };
    // The pool of the command queue the calling thread currently issues on.
    Pool& current_pool();
    std::vector<Pool> pools;  // one per command queue
};

}  // namespace ttml::core::distributed
