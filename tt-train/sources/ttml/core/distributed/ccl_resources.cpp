// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "ccl_resources.hpp"

#include "autograd/auto_context.hpp"

namespace ttml::core::distributed {

CCLResources::CCLResources() {
    auto* device = &ttml::autograd::ctx().get_device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto core_range_set = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange{
        tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{num_cores_x - 1U, num_cores_y - 1U}});

    barrier_semaphores.reserve(kNumSemaphoresPairs);
    all_gather_semaphores.reserve(kNumSemaphoresPairs * kNumSemaphoresPerAllGather);
    reduce_scatter_semaphores.reserve(kNumSemaphoresPairs * kNumSemaphoresPerReduceScatterCall);
    for (uint32_t idx = 0; idx < kNumSemaphoresPairs; ++idx) {
        barrier_semaphores.push_back(
            tt::tt_metal::CreateGlobalSemaphore(device, core_range_set, /* initial_value */ 0));
        for (uint32_t adx = 0; adx < kNumSemaphoresPerAllGather; ++adx) {
            all_gather_semaphores.push_back(
                tt::tt_metal::CreateGlobalSemaphore(device, core_range_set, /* initial_value */ 0));
        }
        for (uint32_t rdx = 0; rdx < kNumSemaphoresPerReduceScatterCall; ++rdx) {
            reduce_scatter_semaphores.push_back(
                tt::tt_metal::CreateGlobalSemaphore(device, core_range_set, /* initial_value */ 0));
        }
    }
}

tt::tt_metal::GlobalSemaphore CCLResources::get_barrier_semaphore() {
    auto barrier_semaphore = barrier_semaphores[barrier_semaphore_index];
    barrier_semaphore_index = (barrier_semaphore_index + 1U) % kNumSemaphoresPairs;
    return barrier_semaphore;
}

std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::get_all_gather_semaphore() {
    std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
    semaphores.reserve(kNumSemaphoresPerAllGather);
    std::copy(
        all_gather_semaphores.begin() + all_gather_semaphore_index,
        all_gather_semaphores.begin() + all_gather_semaphore_index + kNumSemaphoresPerAllGather,
        std::back_inserter(semaphores));
    all_gather_semaphore_index =
        (all_gather_semaphore_index + kNumSemaphoresPerAllGather) % (kNumSemaphoresPairs * kNumSemaphoresPerAllGather);
    return semaphores;
}
std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::get_reduce_scatter_semaphores() {
    std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
    semaphores.reserve(kNumSemaphoresPerReduceScatterCall);
    std::copy(
        reduce_scatter_semaphores.begin() + reduce_scatter_semaphore_index,
        reduce_scatter_semaphores.begin() + reduce_scatter_semaphore_index + kNumSemaphoresPerReduceScatterCall,
        std::back_inserter(semaphores));
    reduce_scatter_semaphore_index = (reduce_scatter_semaphore_index + kNumSemaphoresPerReduceScatterCall) %
                                     (kNumSemaphoresPairs * kNumSemaphoresPerReduceScatterCall);
    return semaphores;
}

}  // namespace ttml::core::distributed
