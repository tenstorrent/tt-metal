// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <stddef.h>
#include <unordered_set>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal {
class IDevice;
using DeviceAddr = uint64_t;
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental::service {

// Manages reservation of free FD dispatch-column cores for long-running service kernels,
// and per-core L1 allocation for those cores.
//
// A "service core" is an FD dispatch-column core that FD has not allocated to its own
// pipeline (prefetcher, dispatcher, fabric-mux, etc.). Services run persistent SD-launched
// kernels on these cores concurrently with FD workloads on the regular worker grid — two
// disjoint worker spaces, no overlap.
//
// Supported configurations: Blackhole or UBB Galaxy, with an active manual FD session
// (initialize_fast_dispatch called, terminate_fast_dispatch not yet called). claim() and
// get_claimable_cores() TT_FATAL outside these conditions.
//
// Not thread-safe. Expected usage is sequential calls from the main thread during
// application setup/teardown.
//
// Runtime flow:
//
//   // 1. Bring up FD, load weights
//   DispatchContext::get().initialize_fast_dispatch(mesh_device);
//   EnqueueMeshWorkload(mesh_cq, weights_workload, true);
//
//   // 2. Claim service cores (must be done while FD is active)
//   auto& svc = ServiceCoreClaims::get();
//   auto claimable = svc.get_claimable_cores(device);
//   svc.claim(device, claimable);
//
//   // 3. Allocate per-core L1 and build the service workload
//   for (auto& core : claimable)
//       counter_addrs[core] = svc.allocate_l1(device, core, sizeof(uint32_t));
//
//   // add_program internally routes to the SD path because cores are claimed.
//   // EnqueueMeshWorkload dispatches service programs via SD, regular programs via FD.
//   // User is responsible for service kernel lifetime (fire and forget).
//   service_workload.add_program(device_range, std::move(service_program));
//   EnqueueMeshWorkload(mesh_cq, service_workload, false);
//
//   // 4. Regular FD workloads run transparently alongside service kernels
//   for (int i = 0; i < N; i++)
//       EnqueueMeshWorkload(mesh_cq, standard_fd_workload, false);
//
//   // 5. Stop service kernel and release cores
//   WriteToDeviceL1(device, stop_core, stop_addr, 1);  // or via GlobalSemaphore
//   svc.release(device, claimable);
//
class ServiceCoreClaims {
public:
    static ServiceCoreClaims& get();

    // Returns dispatch-column cores not yet allocated to FD infra or claimed as service
    // cores. TT_FATALs if no manual FD session is active — the dispatch pool is only in
    // its final post-allocation state after initialize_fast_dispatch() completes, so
    // calling this before that would return a misleading set.
    // Use this to discover available cores before calling claim().
    std::vector<CoreCoord> get_claimable_cores(IDevice* device) const;

    // Reserve one or more free FD-column cores for service use. Constructs a per-core L1
    // allocator for each claimed core. Claimed cores are explicitly removed from
    // dispatch_core_manager's pool on every initialize_fast_dispatch() so FD never
    // allocates them, regardless of how many FD teardown/re-init cycles occur.
    // TT_FATALs if:
    //   - arch is not Blackhole or UBB Galaxy
    //   - no manual FD session is active (initialize_fast_dispatch not yet called)
    //   - any core in the list is already claimed
    void claim(IDevice* device, const std::vector<CoreCoord>& cores);

    // Release one or more claimed cores and destroy their L1 allocators. All addresses
    // handed out by allocate_l1() for these cores become invalid after this call.
    // Silent no-op for unclaimed cores — safe to call in teardown/destructor paths.
    void release(IDevice* device, const std::vector<CoreCoord>& cores);

    // Returns the set of currently claimed cores for a device.
    std::unordered_set<CoreCoord> claimed_cores(ChipId device_id) const;

    // Called from Device::close() to drop all claims for a device.
    void on_device_close(ChipId device_id);

    // Per-core L1 allocator. Valid only for currently-claimed cores; TT_FATALs otherwise.
    // Alignment is fixed at claim() time to HalMemType::DRAM so NoC rd/wr to allocations
    // are always valid (mirrors BankManager's lockstep L1 path). TT_FATALs on OOM.
    // Each service core owns a completely independent L1 range — no interaction with
    // the worker-grid BankManager.
    DeviceAddr allocate_l1(IDevice* device, CoreCoord core, size_t size);
    void deallocate_l1(IDevice* device, CoreCoord core, DeviceAddr addr);
    size_t bytes_available(IDevice* device, CoreCoord core) const;

    // Returns the FD-mode compute grid snapshotted at first claim() for a device, or
    // nullopt if no service cores are currently claimed. Used internally to cap
    // compute_with_storage_grid_size() in SD mode so SD workloads don't accidentally
    // target dispatch-column cores running persistent service kernels — preserving the
    // disjoint worker-set invariant between the regular worker grid and service cores.
    std::optional<CoreCoord> get_safe_compute_grid(ChipId device_id) const;

    // Block until the service kernel on the given core signals completion (RUN_MSG_DONE).
    // Only meaningful for non-persistent kernels that are expected to return; hangs
    // indefinitely if the kernel loops forever.
    void wait_done(IDevice* device, CoreCoord core) const;

    // --- Internal dispatch routing (not user-facing) ---
    // Called on every EnqueueMeshWorkload; must be O(1).
    bool has_any_claims() const;
    // Called at add_program time to route programs targeting service cores to the SD path.
    bool is_service_core(CoreCoord core) const;

    ServiceCoreClaims(const ServiceCoreClaims&) = delete;
    ServiceCoreClaims& operator=(const ServiceCoreClaims&) = delete;

private:
    ServiceCoreClaims();
    ~ServiceCoreClaims();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::experimental::service
