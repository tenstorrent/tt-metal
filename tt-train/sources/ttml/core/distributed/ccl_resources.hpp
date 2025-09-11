// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>

namespace ttml::core::distributed {

class CCLResources {
public:
    CCLResources();

    tt::tt_metal::GlobalSemaphore get_barrier_semaphore();

    std::vector<tt::tt_metal::GlobalSemaphore> get_all_gather_semaphore();

    std::vector<tt::tt_metal::GlobalSemaphore> get_reduce_scatter_semaphores();

private:
    static constexpr uint32_t kNumSemaphoresPairs = 2U;
    static constexpr uint32_t kNumSemaphoresPerAllGather = 2U;
    static constexpr uint32_t kNumSemaphoresPerReduceScatterCall = 3U;

    uint32_t barrier_semaphore_index = 0U;
    uint32_t all_gather_semaphore_index = 0U;
    uint32_t reduce_scatter_semaphore_index = 0U;

    std::vector<tt::tt_metal::GlobalSemaphore> barrier_semaphores;
    std::vector<tt::tt_metal::GlobalSemaphore> all_gather_semaphores;
    std::vector<tt::tt_metal::GlobalSemaphore> reduce_scatter_semaphores;
};

}  // namespace ttml::core::distributed
