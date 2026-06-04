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

namespace tt::tt_metal::internal {

// Internal, unstable API - read the stability/usage conditions in tt_metal/api/internal/README.md
// before depending on anything here.
//
// Manages reservation of free FD dispatch-column cores for long-running service kernels,
// and per-core L1 allocation for those cores.
//
// A "service core" is an FD dispatch-column core that FD has not allocated to its own
// pipeline (prefetcher, dispatcher, fabric-mux, etc.). Services run persistent SD-launched
// kernels on these cores concurrently with FD workloads on the regular worker grid — two
// disjoint worker spaces, no overlap.
//
// Supported configurations: Blackhole or UBB Galaxy, with an active manual FD session.
// claim() and get_claimable_cores() TT_FATAL outside these conditions.
//
// NOTE: not thread-safe api. Expected usage is sequential calls from the main thread during
// application setup/teardown.
//
// Launch once contract: For now, a claimed service core accepts exactly ONE service workload
// enqueue. A second EnqueueMeshWorkload targeting an already launched core TT_FATALs, until the
// core is released - regardless of whether the service kernel on it has actually finished.
// Re-enqueue therefore requires release() then claim() again. This will most likely change in the future.
//
// Runtime flow:
//
//   // 1. Launch App in FD
//
//   // 2. Claim service cores (must be done while FD is active)
//   auto& svc = ServiceCoreManager::get();
//   auto claimable = svc.get_claimable_cores(device);
//   svc.claim(device, claimable);
//
//   // 3. Allocate per-core L1 and build the service workload
//   for (auto& core : claimable)
//       counter_addrs[core] = svc.allocate_l1(device, core, sizeof(uint32_t));
//
//   // add_program just records the program; it has no device handle so it does no routing.
//   // EnqueueMeshWorkload knows the device, so that is where programs targeting claimed service
//   // cores are split off to the SD path while regular programs go via FD.
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
class ServiceCoreManager {
public:
    static ServiceCoreManager& get();

    // Returns dispatch-column cores not yet allocated to FD infra or claimed as service cores.
    // TT_FATALs if:
    //   - no manual FD session is active (the pool is only fully partitioned after
    //     initialize_fast_dispatch() completes; calling before that returns a misleading set)
    //   - no cores are available (all dispatch-column cores are in use by FD or already claimed)
    // Use this immediately before claim() — the query and then claim is the intended idiom.
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
    // Caller contract: the service kernel must already be stopped before release() - the
    // runtime cannot detect completion of a persistent (looping) kernel.
    // TODO: accept an optional user completion predicate (e.g. polls an L1 done-signal that the
    // kernel sets on exit) so release() can verify/wait for termination instead of relying on
    // caller ordering.
    void release(IDevice* device, const std::vector<CoreCoord>& cores);

    // Returns the set of currently claimed cores for a device.
    std::unordered_set<CoreCoord> claimed_cores(ChipId device_id) const;

    // Called from Device::close() to drop all claims for a device.
    void on_device_close(ChipId device_id);

    // Per-core L1 allocator. Valid only for currently claimed cores - TT_FATALs otherwise.
    // Alignment is fixed at claim() time to HalMemType::DRAM so NoC read/write to allocations
    // are always valid (mirrors BankManager's lockstep L1 path). TT_FATALs on OOM.
    // Each service core owns a completely independent L1 range - no interaction with
    // the worker-grid BankManager.
    // Allocates top-down (from L1_END downward) so service buffers and CBs (which grow up
    // from DEFAULT_UNRESERVED) stay in disjoint zones — same convention as worker-core L1 buffers.
    DeviceAddr allocate_l1(IDevice* device, CoreCoord core, size_t size);
    void deallocate_l1(IDevice* device, CoreCoord core, DeviceAddr addr);
    size_t bytes_available(IDevice* device, CoreCoord core) const;
    // Returns the lowest start address currently handed out by the per-core L1 allocator,
    // or nullopt if nothing has been allocated on that core yet.
    // Because allocate_l1() allocates top-down (bottom_up=false), this is the frontier —
    // the lowest point service buffers have reached. Used by validate_circular_buffer_region
    // to detect collisions with CBs that grow up from DEFAULT_UNRESERVED.
    std::optional<DeviceAddr> lowest_allocated_address(ChipId device_id, CoreCoord core) const;

    // Returns the FD-mode compute grid snapshotted at first claim() for a device, or
    // nullopt if no service cores are currently claimed. Used internally to cap
    // compute_with_storage_grid_size() in SD mode so SD workloads don't accidentally
    // target dispatch column cores running persistent service kernels - preserving the
    // disjoint worker-set invariant between the regular worker grid and service cores.
    std::optional<CoreCoord> get_safe_compute_grid(ChipId device_id) const;

    // NOTE: Internal dispatch routing (not user-facing)
    // Called on every EnqueueMeshWorkload
    bool has_any_claims() const;
    // True if `core` is claimed as a service core on `device_id`. Used at enqueue time to route
    // programs to the SD path and to device-scope placement/CB validation.
    bool is_service_core(ChipId device_id, CoreCoord core) const;
    // Enforces launch-once: marks a claimed service core as launched, TT_FATALing if it
    // already was. A claimed core accepts a single service-workload enqueue until release()
    // clears the claim. No-op for cores not claimed on device_id (e.g. worker cores).
    void mark_launched(ChipId device_id, CoreCoord core);

    ServiceCoreManager(const ServiceCoreManager&) = delete;
    ServiceCoreManager& operator=(const ServiceCoreManager&) = delete;

private:
    ServiceCoreManager();
    ~ServiceCoreManager();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::internal
