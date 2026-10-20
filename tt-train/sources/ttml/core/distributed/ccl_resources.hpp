// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <optional>
#include <unordered_map>

namespace ttml::core::distributed {

// Semaphores are pooled per cluster axis. Device groups along different axes are not kept in lockstep
// (e.g. DP replicas drift apart during backward), so a collective on one axis must never reuse a
// semaphore address that a collective on another axis may still be waiting on.
class CCLResources {
public:
    CCLResources() = default;

    tt::tt_metal::GlobalSemaphore get_barrier_semaphore(std::optional<uint32_t> cluster_axis = std::nullopt);

    std::vector<tt::tt_metal::GlobalSemaphore> get_all_gather_semaphore(
        std::optional<uint32_t> cluster_axis = std::nullopt);

    std::vector<tt::tt_metal::GlobalSemaphore> get_reduce_scatter_semaphores(
        std::optional<uint32_t> cluster_axis = std::nullopt);

    std::vector<tt::tt_metal::GlobalSemaphore> get_all_reduce_barrier_semaphores(
        std::optional<uint32_t> cluster_axis = std::nullopt);

private:
    static constexpr uint32_t kNumSemaphoresPairs = 2U;
    static constexpr uint32_t kNumSemaphoresPerAllGather = 2U;
    static constexpr uint32_t kNumSemaphoresPerReduceScatterCall = 3U;
    static constexpr uint32_t kNumSemaphoresPerAllReduceBarrierCall = 2U;

    struct SemaphorePools {
        explicit SemaphorePools(ttnn::distributed::MeshDevice& device);

        uint32_t barrier_semaphore_index = 0U;
        uint32_t all_gather_semaphore_index = 0U;
        uint32_t reduce_scatter_semaphore_index = 0U;
        uint32_t all_reduce_barrier_semaphore_index = 0U;

        std::vector<tt::tt_metal::GlobalSemaphore> barrier_semaphores;
        std::vector<tt::tt_metal::GlobalSemaphore> all_gather_semaphores;
        std::vector<tt::tt_metal::GlobalSemaphore> reduce_scatter_semaphores;
        std::vector<tt::tt_metal::GlobalSemaphore> all_reduce_barrier_semaphores;
    };

    SemaphorePools& get_pools(std::optional<uint32_t> cluster_axis);

    // Key 0 is reserved for calls without a cluster axis; axis `a` maps to key `a + 1`.
    std::unordered_map<uint32_t, SemaphorePools> m_pools;
};

}  // namespace ttml::core::distributed
