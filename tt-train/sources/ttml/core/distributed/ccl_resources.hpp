// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <map>
#include <utility>

namespace ttml::core::distributed {

class CCLResources {
public:
    CCLResources();

    tt::tt_metal::GlobalSemaphore get_barrier_semaphore();

    std::vector<tt::tt_metal::GlobalSemaphore> get_all_gather_semaphore();

    std::vector<tt::tt_metal::GlobalSemaphore> get_reduce_scatter_semaphores();

    std::vector<tt::tt_metal::GlobalSemaphore> get_all_reduce_barrier_semaphores();

    // high_bw_all_gather's {ready, data_valid} pair. One pair serves every call.
    std::pair<tt::tt_metal::GlobalSemaphore, tt::tt_metal::GlobalSemaphore> get_high_bw_all_gather_semaphores();

    // Whether each device on `cluster_axis` is a direct physical neighbor of the next over `num_links` links.
    bool has_direct_neighbor_route(const ttnn::Tensor& tensor, uint32_t cluster_axis, uint32_t num_links);

private:
    static constexpr uint32_t kNumSemaphoresPairs = 2U;
    static constexpr uint32_t kNumSemaphoresPerAllGather = 2U;
    static constexpr uint32_t kNumSemaphoresPerReduceScatterCall = 3U;
    static constexpr uint32_t kNumSemaphoresPerAllReduceBarrierCall = 2U;

    uint32_t barrier_semaphore_index = 0U;
    uint32_t all_gather_semaphore_index = 0U;
    uint32_t reduce_scatter_semaphore_index = 0U;
    uint32_t all_reduce_barrier_semaphore_index = 0U;

    std::vector<tt::tt_metal::GlobalSemaphore> barrier_semaphores;
    std::vector<tt::tt_metal::GlobalSemaphore> all_gather_semaphores;
    std::vector<tt::tt_metal::GlobalSemaphore> reduce_scatter_semaphores;
    std::vector<tt::tt_metal::GlobalSemaphore> all_reduce_barrier_semaphores;
    std::pair<tt::tt_metal::GlobalSemaphore, tt::tt_metal::GlobalSemaphore> high_bw_all_gather_semaphores;
    std::map<std::pair<uint32_t, uint32_t>, bool> direct_neighbor_routes;
};

}  // namespace ttml::core::distributed
