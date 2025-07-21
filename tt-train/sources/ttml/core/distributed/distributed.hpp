// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/host_api.hpp>

#include "autograd/auto_context.hpp"
#include "serialization/serializable.hpp"

namespace ttml::core::distributed {

using Rank = tt::tt_metal::distributed::multihost::Rank;
using Tag = tt::tt_metal::distributed::multihost::Tag;

struct CCLResources {
    static constexpr uint32_t kNumSemaphoresPairs = 2U;
    static constexpr uint32_t kNumSemaphoresPerReduceScatterCall = 3U;

    CCLResources() {
        auto* device = &ttml::autograd::ctx().get_device();
        auto core_range_set = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange{0, 0}, tt::tt_metal::CoreRange{7, 6});

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

    tt::tt_metal::GlobalSemaphore& get_all_gather_semaphore() {
        auto& semaphore = all_gather_semaphores[all_gather_semaphore_index];
        all_gather_semaphore_index = (all_gather_semaphore_index + 1U) % kNumSemaphoresPairs;
        return semaphore;
    }

    std::vector<tt::tt_metal::GlobalSemaphore> get_reduce_scatter_semaphores() {
        std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
        semaphores.reserve(kNumSemaphoresPerReduceScatterCall);
        for (uint32_t idx = 0; idx < kNumSemaphoresPerReduceScatterCall; ++idx) {
            semaphores.push_back(reduce_scatter_semaphores[reduce_scatter_semaphore_index + idx]);
        }
        reduce_scatter_semaphore_index = (reduce_scatter_semaphore_index + kNumSemaphoresPerReduceScatterCall) %
                                         (kNumSemaphoresPairs * kNumSemaphoresPerReduceScatterCall);
        return semaphores;
    }

    uint32_t all_gather_semaphore_index = 0U;
    uint32_t reduce_scatter_semaphore_index = 0U;

    std::vector<tt::tt_metal::GlobalSemaphore> all_gather_semaphores;
    std::vector<tt::tt_metal::GlobalSemaphore> reduce_scatter_semaphores;
};

ttnn::Tensor synchronize_tensor_fabric(const ttnn::Tensor& tensor, CCLResources& ccl_resources);

ttnn::Tensor synchronize_tensor(const ttnn::Tensor& tensor);
void synchronize_parameters(const serialization::NamedParameters& parameters);

void send_tensor(const autograd::DistributedContext& ctx, const ttnn::Tensor& tensor, Rank dest, Tag tag = Tag{0});
void recv_tensor(const autograd::DistributedContext& ctx, ttnn::Tensor& tensor, Rank source, Tag tag = Tag{0});
void broadcast_tensor(const autograd::DistributedContext& ctx, ttnn::Tensor& tensor, Rank root);

}  // namespace ttml::core::distributed
