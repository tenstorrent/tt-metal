// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>

namespace ttml::core::distributed {

// Semaphores are handed out round-robin from small pools, so a collective reuses the semaphores of an
// earlier one. That is only safe between collectives over the same device groups, so each instance must
// serve a single cluster axis (see AutoContext::get_ccl_resources). Groups along different axes are not
// kept in lockstep: a device that is ahead on one axis would signal a lagging peer on a semaphore that
// the peer is still using for a collective on the other axis.
class CCLResources {
public:
    CCLResources();

    tt::tt_metal::GlobalSemaphore get_barrier_semaphore();

    std::vector<tt::tt_metal::GlobalSemaphore> get_all_gather_semaphore();

    std::vector<tt::tt_metal::GlobalSemaphore> get_reduce_scatter_semaphores();

    std::vector<tt::tt_metal::GlobalSemaphore> get_all_reduce_barrier_semaphores();

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
};

}  // namespace ttml::core::distributed
