// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace ttml::core::distributed {

// Device-resident resources the collective wrappers in ttnn_fixed::distributed reuse across calls:
// global semaphores and the reduce-scatter staging buffers.
//
// Collectives are cross-device operations. When device E starts a collective it writes into every
// peer's copy of the semaphores and staging buffers, while a slower peer D may still be finishing
// the previous collective on the same resources. Both kinds of resource therefore rotate through
// several sets, and the rotation depth is the slack that covers the skew between devices. Each
// command queue gets its own semaphore pool: collectives issued on different queues can be in flight
// at the same time (FSDP gathers on the second queue next to a tensor-parallel all-reduce on the
// first) and must not rotate through shared semaphores.
class CCLResources {
public:
    CCLResources();

    tt::tt_metal::GlobalSemaphore get_barrier_semaphore();
    std::vector<tt::tt_metal::GlobalSemaphore> get_all_gather_semaphore();
    std::vector<tt::tt_metal::GlobalSemaphore> get_reduce_scatter_semaphores();
    std::vector<tt::tt_metal::GlobalSemaphore> get_all_reduce_barrier_semaphores();

    // Persistent staging buffers for reduce_scatter_minimal_async's contiguous ring path.
    //
    // Without them the op allocates an intermediate and a "penult" tensor per call and frees both
    // on return; each is a mesh-wide allocation whose host cost grows with the device count
    // (~100 us per call on 8 chips, ~350 us on 32), and on a second command queue the freed
    // addresses can be handed to compute while the collective is still using them. Returns the
    // next of kNumStagingSets sets for this (queue, input spec, dim, axis, topology), or an empty
    // vector when the configuration has no contiguous-path staging layout (Linear topology, dim 0,
    // 2-device rings) and the op has to allocate for itself.
    const std::vector<ttnn::Tensor>& get_reduce_scatter_staging_buffers(
        const ttnn::Tensor& input, int dim, std::optional<uint32_t> cluster_axis, ttnn::ccl::Topology topology);

private:
    static constexpr uint32_t kNumSemaphoreSets = 8U;
    static constexpr uint32_t kNumStagingSets = 4U;
    static constexpr uint32_t kNumSemaphoresPerAllGather = 2U;
    static constexpr uint32_t kNumSemaphoresPerReduceScatterCall = 3U;
    static constexpr uint32_t kNumSemaphoresPerAllReduceBarrierCall = 2U;

    // kNumSemaphoreSets groups of `per_call` semaphores handed out round-robin.
    struct SemaphoreRing {
        std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
        uint32_t per_call = 1U;
        uint32_t next = 0U;
        std::vector<tt::tt_metal::GlobalSemaphore> take();
    };
    struct Pool {
        SemaphoreRing barrier;
        SemaphoreRing all_gather;
        SemaphoreRing reduce_scatter;
        SemaphoreRing all_reduce_barrier;
    };
    Pool& current_pool();
    std::vector<Pool> pools;  // one per command queue

    struct StagingRotation {
        std::vector<std::vector<ttnn::Tensor>> sets;
        uint32_t next = 0U;
    };
    std::unordered_map<std::string, StagingRotation> reduce_scatter_staging_buffers;
};

}  // namespace ttml::core::distributed
