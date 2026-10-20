// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ccl_resources.hpp"

#include "autograd/auto_context.hpp"

namespace ttml::core::distributed {

namespace {

std::vector<tt::tt_metal::GlobalSemaphore> take_semaphores(
    const std::vector<tt::tt_metal::GlobalSemaphore>& pool, uint32_t& index, uint32_t count) {
    std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
    semaphores.reserve(count);
    std::copy(pool.begin() + index, pool.begin() + index + count, std::back_inserter(semaphores));
    index = (index + count) % static_cast<uint32_t>(pool.size());
    return semaphores;
}

}  // namespace

CCLResources::SemaphorePools::SemaphorePools(ttnn::distributed::MeshDevice& device) {
    auto compute_with_storage_grid_size = device.compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto core_range_set = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange{
        tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{num_cores_x - 1U, num_cores_y - 1U}});

    barrier_semaphores.reserve(kNumSemaphoresPairs);
    all_gather_semaphores.reserve(kNumSemaphoresPairs * kNumSemaphoresPerAllGather);
    reduce_scatter_semaphores.reserve(kNumSemaphoresPairs * kNumSemaphoresPerReduceScatterCall);
    all_reduce_barrier_semaphores.reserve(kNumSemaphoresPairs * kNumSemaphoresPerAllReduceBarrierCall);
    for (uint32_t idx = 0; idx < kNumSemaphoresPairs; ++idx) {
        barrier_semaphores.push_back(tt::tt_metal::GlobalSemaphore(device, core_range_set, /* initial_value */ 0));
        for (uint32_t adx = 0; adx < kNumSemaphoresPerAllGather; ++adx) {
            all_gather_semaphores.push_back(
                tt::tt_metal::GlobalSemaphore(device, core_range_set, /* initial_value */ 0));
        }
        for (uint32_t rdx = 0; rdx < kNumSemaphoresPerReduceScatterCall; ++rdx) {
            reduce_scatter_semaphores.push_back(
                tt::tt_metal::GlobalSemaphore(device, core_range_set, /* initial_value */ 0));
        }
        for (uint32_t rdx = 0; rdx < kNumSemaphoresPerAllReduceBarrierCall; ++rdx) {
            all_reduce_barrier_semaphores.push_back(
                tt::tt_metal::GlobalSemaphore(device, core_range_set, /* initial_value */ 0));
        }
    }
}

CCLResources::SemaphorePools& CCLResources::get_pools(std::optional<uint32_t> cluster_axis) {
    const uint32_t key = cluster_axis.has_value() ? cluster_axis.value() + 1U : 0U;
    auto it = m_pools.find(key);
    if (it == m_pools.end()) {
        it = m_pools.emplace(key, SemaphorePools(ttml::autograd::ctx().get_device())).first;
    }
    return it->second;
}

tt::tt_metal::GlobalSemaphore CCLResources::get_barrier_semaphore(std::optional<uint32_t> cluster_axis) {
    auto& pools = get_pools(cluster_axis);
    return take_semaphores(pools.barrier_semaphores, pools.barrier_semaphore_index, 1U).front();
}

std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::get_all_gather_semaphore(
    std::optional<uint32_t> cluster_axis) {
    auto& pools = get_pools(cluster_axis);
    return take_semaphores(pools.all_gather_semaphores, pools.all_gather_semaphore_index, kNumSemaphoresPerAllGather);
}

std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::get_reduce_scatter_semaphores(
    std::optional<uint32_t> cluster_axis) {
    auto& pools = get_pools(cluster_axis);
    return take_semaphores(
        pools.reduce_scatter_semaphores, pools.reduce_scatter_semaphore_index, kNumSemaphoresPerReduceScatterCall);
}

std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::get_all_reduce_barrier_semaphores(
    std::optional<uint32_t> cluster_axis) {
    auto& pools = get_pools(cluster_axis);
    return take_semaphores(
        pools.all_reduce_barrier_semaphores,
        pools.all_reduce_barrier_semaphore_index,
        kNumSemaphoresPerAllReduceBarrierCall);
}

}  // namespace ttml::core::distributed
