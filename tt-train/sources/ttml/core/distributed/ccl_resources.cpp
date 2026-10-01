// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ccl_resources.hpp"

#include <fmt/format.h>

#include "autograd/auto_context.hpp"
#include "ttnn/core.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async.hpp"

namespace ttml::core::distributed {

std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::SemaphoreRing::take() {
    std::vector<tt::tt_metal::GlobalSemaphore> out(semaphores.begin() + next, semaphores.begin() + next + per_call);
    next = (next + per_call) % static_cast<uint32_t>(semaphores.size());
    return out;
}

CCLResources::CCLResources() {
    auto& ctx = ttml::autograd::ctx();
    auto& device = ctx.get_device();

    // The whole chip, not the compute rectangle: with a CCL sub-device the collective kernels run on
    // the reserved cores and must find the semaphores there as well.
    const auto grid = ctx.full_compute_grid_size();
    const auto all_cores = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange{
        tt::tt_metal::CoreCoord{0, 0},
        tt::tt_metal::CoreCoord{static_cast<uint32_t>(grid.x) - 1U, static_cast<uint32_t>(grid.y) - 1U}});

    const auto make_ring = [&](uint32_t per_call) {
        SemaphoreRing ring;
        ring.per_call = per_call;
        ring.semaphores.reserve(static_cast<size_t>(per_call) * kNumSemaphoreSets);
        for (uint32_t i = 0; i < per_call * kNumSemaphoreSets; ++i) {
            ring.semaphores.emplace_back(device, all_cores, /* initial_value */ 0);
        }
        return ring;
    };
    pools.reserve(ctx.num_command_queues());
    for (size_t queue = 0; queue < ctx.num_command_queues(); ++queue) {
        pools.push_back(Pool{
            .barrier = make_ring(1U),
            .all_gather = make_ring(kNumSemaphoresPerAllGather),
            .reduce_scatter = make_ring(kNumSemaphoresPerReduceScatterCall),
            .all_reduce_barrier = make_ring(kNumSemaphoresPerAllReduceBarrierCall),
        });
    }
}

CCLResources::Pool& CCLResources::current_pool() {
    const auto queue = static_cast<size_t>(*ttnn::core::get_current_command_queue_id_for_thread());
    TT_FATAL(
        queue < pools.size(),
        "CCLResources: collective issued on command queue {} but the device was opened with {} queue(s)",
        queue,
        pools.size());
    return pools[queue];
}

tt::tt_metal::GlobalSemaphore CCLResources::get_barrier_semaphore() {
    return current_pool().barrier.take().front();
}

std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::get_all_gather_semaphore() {
    return current_pool().all_gather.take();
}

std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::get_reduce_scatter_semaphores() {
    return current_pool().reduce_scatter.take();
}

std::vector<tt::tt_metal::GlobalSemaphore> CCLResources::get_all_reduce_barrier_semaphores() {
    return current_pool().all_reduce_barrier.take();
}

const std::vector<ttnn::Tensor>& CCLResources::get_reduce_scatter_staging_buffers(
    const ttnn::Tensor& input, int dim, std::optional<uint32_t> cluster_axis, ttnn::ccl::Topology topology) {
    // Everything the staging spec depends on, plus the queue: a reduce-scatter on each queue can be
    // in flight at the same time and they must not share staging memory.
    const auto& mem = input.memory_config();
    const auto key = fmt::format(
        "q{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
        static_cast<uint32_t>(*ttnn::core::get_current_command_queue_id_for_thread()),
        input.logical_shape(),
        input.padded_shape(),
        static_cast<int>(input.dtype()),
        static_cast<int>(input.layout()),
        static_cast<int>(mem.buffer_type()),
        static_cast<int>(mem.memory_layout()),
        dim,
        cluster_axis.has_value() ? static_cast<int>(*cluster_axis) : -1,
        static_cast<int>(topology));

    auto it = reduce_scatter_staging_buffers.find(key);
    if (it == reduce_scatter_staging_buffers.end()) {
        StagingRotation rotation;
        for (uint32_t set = 0; set < kNumStagingSets; ++set) {
            std::vector<ttnn::Tensor> buffers;
            try {
                buffers = ttnn::experimental::reduce_scatter_minimal_async_create_intermediate_buffer(
                    input, dim, topology, cluster_axis, /* compute_kernel_config */ std::nullopt);
            } catch (const std::exception&) {
                buffers.clear();  // no contiguous staging layout for this configuration
            }
            rotation.sets.push_back(std::move(buffers));
            if (rotation.sets.back().empty()) {
                break;  // one empty set is enough to say "not applicable"
            }
        }
        it = reduce_scatter_staging_buffers.emplace(key, std::move(rotation)).first;
    }
    auto& rotation = it->second;
    const auto& buffers = rotation.sets[rotation.next % rotation.sets.size()];
    rotation.next = (rotation.next + 1U) % static_cast<uint32_t>(rotation.sets.size());
    return buffers;
}

}  // namespace ttml::core::distributed
