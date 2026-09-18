// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ccl_resources.hpp"

#include "autograd/auto_context.hpp"
#include "ttnn/core.hpp"

namespace ttml::core::distributed {

std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::SemaphoreRing::take() {
    std::vector<tt::tt_metal::GlobalSemaphore> out(semaphores.begin() + next, semaphores.begin() + next + per_call);
    next = (next + per_call) % static_cast<uint32_t>(semaphores.size());
    return out;
}

CCLResources::CCLResources() {
    auto& ctx = ttml::autograd::ctx();
    auto& device = ctx.get_device();

    // The whole chip, not the compute rectangle: with a CCL sub-device the collective kernels run on
    // the reserved cores and must find the semaphores there as well.
    const auto grid = ctx.full_compute_grid_size();
    const auto all_cores = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange{
        tt::tt_metal::CoreCoord{0, 0},
        tt::tt_metal::CoreCoord{static_cast<uint32_t>(grid.x) - 1U, static_cast<uint32_t>(grid.y) - 1U}});

    const auto make_ring = [&](uint32_t per_call) {
        SemaphoreRing ring;
        ring.per_call = per_call;
        ring.semaphores.reserve(static_cast<size_t>(per_call) * kNumSemaphoreSets);
        for (uint32_t i = 0; i < per_call * kNumSemaphoreSets; ++i) {
            ring.semaphores.emplace_back(device, all_cores, /* initial_value */ 0);
        }
        return ring;
    };
    pools.reserve(ctx.num_command_queues());
    for (size_t queue = 0; queue < ctx.num_command_queues(); ++queue) {
        pools.push_back(Pool{
            .barrier = make_ring(1U),
            .all_gather = make_ring(kNumSemaphoresPerAllGather),
            .reduce_scatter = make_ring(kNumSemaphoresPerReduceScatterCall),
            .all_reduce_barrier = make_ring(kNumSemaphoresPerAllReduceBarrierCall),
        });
    }
}

CCLResources::Pool& CCLResources::current_pool() {
    const auto queue = static_cast<size_t>(*ttnn::core::get_current_command_queue_id_for_thread());
    TT_FATAL(
        queue < pools.size(),
        "CCLResources: collective issued on command queue {} but the device was opened with {} queue(s)",
        queue,
        pools.size());
    return pools[queue];
}

tt::tt_metal::GlobalSemaphore CCLResources::get_barrier_semaphore() {
    return current_pool().barrier.take().front();
}

std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::get_all_gather_semaphore() {
    return current_pool().all_gather.take();
}

std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::get_reduce_scatter_semaphores() {
    return current_pool().reduce_scatter.take();
}

std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::get_all_reduce_barrier_semaphores() {
    return current_pool().all_reduce_barrier.take();
}

}  // namespace ttml::core::distributed
