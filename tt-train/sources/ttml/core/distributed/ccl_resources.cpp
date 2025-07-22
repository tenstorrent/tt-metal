#include "ccl_resources.hpp"

#include "autograd/auto_context.hpp"

namespace ttml::core::distributed {

CCLResources::CCLResources() {
    auto* device = &ttml::autograd::ctx().get_device();
    auto core_range_set = tt::tt_metal::CoreRangeSet(
        tt::tt_metal::CoreRange{tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{7, 6}});

    all_gather_semaphores.reserve(kNumSemaphoresPairs);
    reduce_scatter_semaphores.reserve(kNumSemaphoresPairs * kNumSemaphoresPerReduceScatterCall);
    for (uint32_t idx = 0; idx < kNumSemaphoresPairs; ++idx) {
        all_gather_semaphores.push_back(
            tt::tt_metal::CreateGlobalSemaphore(device, core_range_set, /* initial_value */ 0));
        for (uint32_t rdx = 0; rdx < kNumSemaphoresPerReduceScatterCall; ++rdx) {
            reduce_scatter_semaphores.push_back(
                tt::tt_metal::CreateGlobalSemaphore(device, core_range_set, /* initial_value */ 0));
        }
    }
}
tt::tt_metal::GlobalSemaphore& CCLResources::get_all_gather_semaphore() {
    auto& semaphore = all_gather_semaphores[all_gather_semaphore_index];
    all_gather_semaphore_index = (all_gather_semaphore_index + 1U) % kNumSemaphoresPairs;
    return semaphore;
}
std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::get_reduce_scatter_semaphores() {
    std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
    semaphores.reserve(kNumSemaphoresPerReduceScatterCall);
    for (uint32_t idx = 0; idx < kNumSemaphoresPerReduceScatterCall; ++idx) {
        semaphores.push_back(reduce_scatter_semaphores[reduce_scatter_semaphore_index + idx]);
    }
    reduce_scatter_semaphore_index = (reduce_scatter_semaphore_index + kNumSemaphoresPerReduceScatterCall) %
                                     (kNumSemaphoresPairs * kNumSemaphoresPerReduceScatterCall);
    return semaphores;
}

}  // namespace ttml::core::distributed
