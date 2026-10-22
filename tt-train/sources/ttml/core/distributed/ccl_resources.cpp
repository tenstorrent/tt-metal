// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ccl_resources.hpp"

#include "autograd/auto_context.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/mesh_ring_plan.hpp"

namespace ttml::core::distributed {

namespace {

tt::tt_metal::CoreRangeSet make_full_grid_core_range_set(ttnn::distributed::MeshDevice& device) {
    auto compute_with_storage_grid_size = device.compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    return tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange{
        tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{num_cores_x - 1U, num_cores_y - 1U}});
}

std::pair<tt::tt_metal::GlobalSemaphore, tt::tt_metal::GlobalSemaphore> make_high_bw_all_gather_semaphores() {
    auto& device = ttml::autograd::ctx().get_device();
    const auto core_range_set = make_full_grid_core_range_set(device);

    // high_bw_all_gather requires its semaphores in L1_SMALL when the device reserves that region, else in L1.
    const auto buffer_type = device.allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL) > 0
                                 ? tt::tt_metal::BufferType::L1_SMALL
                                 : tt::tt_metal::BufferType::L1;
    return {
        tt::tt_metal::GlobalSemaphore(device, core_range_set, /* initial_value */ 0, buffer_type),
        tt::tt_metal::GlobalSemaphore(device, core_range_set, /* initial_value */ 0, buffer_type)};
}

}  // namespace

CCLResources::CCLResources() {
    auto& device = ttml::autograd::ctx().get_device();
    const auto core_range_set = make_full_grid_core_range_set(device);

    barrier_semaphores.reserve(kNumSemaphoresPairs);
    all_gather_semaphores.reserve(kNumSemaphoresPairs * kNumSemaphoresPerAllGather);
    reduce_scatter_semaphores.reserve(kNumSemaphoresPairs * kNumSemaphoresPerReduceScatterCall);
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

std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::get_all_reduce_barrier_semaphores() {
    std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
    semaphores.reserve(kNumSemaphoresPerAllReduceBarrierCall);
    std::copy(
        all_reduce_barrier_semaphores.begin() + all_reduce_barrier_semaphore_index,
        all_reduce_barrier_semaphores.begin() + all_reduce_barrier_semaphore_index +
            kNumSemaphoresPerAllReduceBarrierCall,
        std::back_inserter(semaphores));
    all_reduce_barrier_semaphore_index = (all_reduce_barrier_semaphore_index + kNumSemaphoresPerAllReduceBarrierCall) %
                                         (kNumSemaphoresPairs * kNumSemaphoresPerAllReduceBarrierCall);
    return semaphores;
}

const std::pair<tt::tt_metal::GlobalSemaphore, tt::tt_metal::GlobalSemaphore>&
CCLResources::get_high_bw_all_gather_semaphores() {
    if (!high_bw_all_gather_semaphores.has_value()) {
        high_bw_all_gather_semaphores.emplace(make_high_bw_all_gather_semaphores());
    }
    return *high_bw_all_gather_semaphores;
}

bool CCLResources::has_direct_neighbor_route(
    const ttnn::Tensor& tensor, const uint32_t cluster_axis, const uint32_t num_links) {
    const auto key = std::pair{cluster_axis, num_links};
    if (const auto it = direct_neighbor_routes.find(key); it != direct_neighbor_routes.end()) {
        return it->second;
    }
    const auto topology = ttnn::ccl::get_axis_topology(tensor, tt::tt_fabric::GetFabricConfig(), cluster_axis);
    const bool has_route = ttnn::operations::ccl::common::resolve_direct_neighbor_route_hash(
                               tensor,
                               cluster_axis,
                               num_links,
                               topology,
                               ttnn::ccl::snake_ring::Orientation::Row,
                               /* log_rejection */ false)
                               .has_value();
    direct_neighbor_routes.emplace(key, has_route);
    return has_route;
}

}  // namespace ttml::core::distributed
