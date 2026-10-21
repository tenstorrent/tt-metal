// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <map>
#include <optional>

namespace ttml::core::distributed {

// Semaphores for the asynchronous CCLs, cycled so consecutive calls use different ones.
//
// Every mesh axis (and the whole mesh, std::nullopt) cycles through its own set. Within one group of
// devices a collective cannot finish on any member before all members have started it, so members are
// at most one call apart and two sets suffice. Across axes there is no such bound: a device that ran
// ahead on one axis can signal a neighbour that is still inside an earlier collective over another
// axis, and with a shared set that collective's reset swallows the signal and the later one hangs
// (seen as a tp reduce-scatter hanging right after a dp gradient all-reduce on a 2x4 mesh).
class CCLResources {
public:
    CCLResources();

    tt::tt_metal::GlobalSemaphore get_barrier_semaphore(std::optional<uint32_t> cluster_axis);

    std::vector<tt::tt_metal::GlobalSemaphore> get_all_gather_semaphore(std::optional<uint32_t> cluster_axis);

    std::vector<tt::tt_metal::GlobalSemaphore> get_reduce_scatter_semaphores(std::optional<uint32_t> cluster_axis);

    std::vector<tt::tt_metal::GlobalSemaphore> get_all_reduce_barrier_semaphores(std::optional<uint32_t> cluster_axis);

private:
    static constexpr uint32_t kNumSemaphoresPairs = 2U;
    static constexpr uint32_t kNumSemaphoresPerAllGather = 2U;
    static constexpr uint32_t kNumSemaphoresPerReduceScatterCall = 3U;
    static constexpr uint32_t kNumSemaphoresPerAllReduceBarrierCall = 2U;

    struct SemaphorePool {
        uint32_t barrier_semaphore_index = 0U;
        uint32_t all_gather_semaphore_index = 0U;
        uint32_t reduce_scatter_semaphore_index = 0U;
        uint32_t all_reduce_barrier_semaphore_index = 0U;

        std::vector<tt::tt_metal::GlobalSemaphore> barrier_semaphores;
        std::vector<tt::tt_metal::GlobalSemaphore> all_gather_semaphores;
        std::vector<tt::tt_metal::GlobalSemaphore> reduce_scatter_semaphores;
        std::vector<tt::tt_metal::GlobalSemaphore> all_reduce_barrier_semaphores;
    };

    SemaphorePool& pool(std::optional<uint32_t> cluster_axis);

    std::map<std::optional<uint32_t>, SemaphorePool> m_pools;
};

}  // namespace ttml::core::distributed
