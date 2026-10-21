// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ccl_resources.hpp"

#include "autograd/auto_context.hpp"

namespace ttml::core::distributed {

CCLResources::CCLResources() {
    auto& device = ttml::autograd::ctx().get_device();

    auto compute_with_storage_grid_size = device.compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto core_range_set = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange{
        tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{num_cores_x - 1U, num_cores_y - 1U}});

    // One pool for the whole mesh and one per axis, all allocated up front.
    std::vector<std::optional<uint32_t>> cluster_axes{std::nullopt};
    for (uint32_t axis = 0; axis < device.shape().dims(); ++axis) {
        cluster_axes.emplace_back(axis);
    }
    for (const auto& cluster_axis : cluster_axes) {
        auto& p = m_pools[cluster_axis];
        p.barrier_semaphores.reserve(kNumSemaphoresPairs);
        p.all_gather_semaphores.reserve(kNumSemaphoresPairs * kNumSemaphoresPerAllGather);
        p.reduce_scatter_semaphores.reserve(kNumSemaphoresPairs * kNumSemaphoresPerReduceScatterCall);
        p.all_reduce_barrier_semaphores.reserve(kNumSemaphoresPairs * kNumSemaphoresPerAllReduceBarrierCall);
        for (uint32_t idx = 0; idx < kNumSemaphoresPairs; ++idx) {
            p.barrier_semaphores.push_back(
                tt::tt_metal::GlobalSemaphore(device, core_range_set, /* initial_value */ 0));
            for (uint32_t adx = 0; adx < kNumSemaphoresPerAllGather; ++adx) {
                p.all_gather_semaphores.push_back(
                    tt::tt_metal::GlobalSemaphore(device, core_range_set, /* initial_value */ 0));
            }
            for (uint32_t rdx = 0; rdx < kNumSemaphoresPerReduceScatterCall; ++rdx) {
                p.reduce_scatter_semaphores.push_back(
                    tt::tt_metal::GlobalSemaphore(device, core_range_set, /* initial_value */ 0));
            }
            for (uint32_t rdx = 0; rdx < kNumSemaphoresPerAllReduceBarrierCall; ++rdx) {
                p.all_reduce_barrier_semaphores.push_back(
                    tt::tt_metal::GlobalSemaphore(device, core_range_set, /* initial_value */ 0));
            }
        }
    }
}

CCLResources::SemaphorePool& CCLResources::pool(std::optional<uint32_t> cluster_axis) {
    auto it = m_pools.find(cluster_axis);
    TT_FATAL(it != m_pools.end(), "No CCL semaphores for cluster axis {}", cluster_axis.value_or(0U));
    return it->second;
}

tt::tt_metal::GlobalSemaphore CCLResources::get_barrier_semaphore(std::optional<uint32_t> cluster_axis) {
    auto& p = pool(cluster_axis);
    auto barrier_semaphore = p.barrier_semaphores[p.barrier_semaphore_index];
    p.barrier_semaphore_index = (p.barrier_semaphore_index + 1U) % kNumSemaphoresPairs;
    return barrier_semaphore;
}

std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::get_all_gather_semaphore(
    std::optional<uint32_t> cluster_axis) {
    auto& p = pool(cluster_axis);
    std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
    semaphores.reserve(kNumSemaphoresPerAllGather);
    std::copy(
        p.all_gather_semaphores.begin() + p.all_gather_semaphore_index,
        p.all_gather_semaphores.begin() + p.all_gather_semaphore_index + kNumSemaphoresPerAllGather,
        std::back_inserter(semaphores));
    p.all_gather_semaphore_index = (p.all_gather_semaphore_index + kNumSemaphoresPerAllGather) %
                                   (kNumSemaphoresPairs * kNumSemaphoresPerAllGather);
    return semaphores;
}
std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::get_reduce_scatter_semaphores(
    std::optional<uint32_t> cluster_axis) {
    auto& p = pool(cluster_axis);
    std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
    semaphores.reserve(kNumSemaphoresPerReduceScatterCall);
    std::copy(
        p.reduce_scatter_semaphores.begin() + p.reduce_scatter_semaphore_index,
        p.reduce_scatter_semaphores.begin() + p.reduce_scatter_semaphore_index + kNumSemaphoresPerReduceScatterCall,
        std::back_inserter(semaphores));
    p.reduce_scatter_semaphore_index = (p.reduce_scatter_semaphore_index + kNumSemaphoresPerReduceScatterCall) %
                                       (kNumSemaphoresPairs * kNumSemaphoresPerReduceScatterCall);
    return semaphores;
}

std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::get_all_reduce_barrier_semaphores(
    std::optional<uint32_t> cluster_axis) {
    auto& p = pool(cluster_axis);
    std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
    semaphores.reserve(kNumSemaphoresPerAllReduceBarrierCall);
    std::copy(
        p.all_reduce_barrier_semaphores.begin() + p.all_reduce_barrier_semaphore_index,
        p.all_reduce_barrier_semaphores.begin() + p.all_reduce_barrier_semaphore_index +
            kNumSemaphoresPerAllReduceBarrierCall,
        std::back_inserter(semaphores));
    p.all_reduce_barrier_semaphore_index =
        (p.all_reduce_barrier_semaphore_index + kNumSemaphoresPerAllReduceBarrierCall) %
        (kNumSemaphoresPairs * kNumSemaphoresPerAllReduceBarrierCall);
    return semaphores;
}

}  // namespace ttml::core::distributed
